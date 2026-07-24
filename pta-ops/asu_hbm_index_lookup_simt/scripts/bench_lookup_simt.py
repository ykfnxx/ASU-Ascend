#!/usr/bin/env python3
"""Benchmark the Ascend 950 SIMT lookup operator with controlled hit counts."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from lookup_simt_common import (
    PKG_DIR,
    assert_runtime_result,
    call_lookup,
    estimate_state_bytes,
    expected_result,
    load_extension,
    require_runtime,
    to_npu_state,
)

from python.random_workload import (  # type: ignore[import-not-found]
    QUERY_COUNT,
    make_random_case,
    validate_hit_count,
)


DEFAULT_HIT_COUNT = 1843


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Ascend 950 SIMT lookup/allocation/LRU eviction. "
            "Every invocation uses a fresh state with an exact hit count; "
            "unique miss token IDs and their query positions are randomized."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=PKG_DIR / "build")
    parser.add_argument("--module-path", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--batch-rounds",
        type=int,
        default=10,
        help=(
            "timed states preloaded per batch; 0 preloads all rounds "
            "(default: 10)"
        ),
    )
    parser.add_argument(
        "--hit-count",
        type=int,
        default=DEFAULT_HIT_COUNT,
        help=f"exact hits per {QUERY_COUNT}-token request (default: {DEFAULT_HIT_COUNT})",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional path for a machine-readable benchmark summary",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.req_num <= 0:
        raise ValueError("--req-num must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.batch_rounds < 0:
        raise ValueError("--batch-rounds cannot be negative")
    if args.device < 0:
        raise ValueError("--device cannot be negative")
    validate_hit_count(args.hit_count)


def verify_one_state(
    np: Any,
    torch: Any,
    module: Any,
    req_num: int,
    hit_count: int,
    seed: int,
) -> None:
    case = make_random_case(
        np,
        req_num,
        hit_count,
        seed=seed,
        case_id=0,
    )
    expected = expected_result(np, case)
    state = to_npu_state(torch, module, case)
    outputs = call_lookup(module, state, req_num)
    torch.npu.synchronize()
    assert_runtime_result(np, state, outputs, expected)


def preload_states(
    np: Any,
    torch: Any,
    module: Any,
    req_num: int,
    hit_count: int,
    count: int,
    seed: int,
    first_case_id: int,
) -> list[Any]:
    states = [
        to_npu_state(
            torch,
            module,
            make_random_case(
                np,
                req_num,
                hit_count,
                seed=seed,
                case_id=first_case_id + offset,
            ),
        )
        for offset in range(count)
    ]
    torch.npu.synchronize()
    return states


def warmup_states(
    torch: Any,
    module: Any,
    states: list[Any],
    req_num: int,
) -> None:
    outputs = [call_lookup(module, state, req_num) for state in states]
    torch.npu.synchronize()
    if len(outputs) != len(states):
        raise RuntimeError("warmup output retention failed")


def run_states(
    torch: Any,
    module: Any,
    states: list[Any],
    req_num: int,
) -> tuple[float, float]:
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    outputs = []

    torch.npu.synchronize()
    start_event.record()
    wall_start = time.perf_counter()
    for state in states:
        outputs.append(call_lookup(module, state, req_num))
    end_event.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    event_ms = float(start_event.elapsed_time(end_event))
    if len(outputs) != len(states):
        raise RuntimeError("timed output retention failed")
    return wall_ms, event_ms


def write_json(path: Path, summary: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    np, torch, _ = require_runtime(args.device)
    module, module_path = load_extension(args.module_path, args.build_dir)
    miss_count = QUERY_COUNT - args.hit_count
    batch_rounds = args.batch_rounds or args.rounds
    batch_rounds = min(batch_rounds, args.rounds)
    state_mib = estimate_state_bytes(args.req_num) / 1024.0 / 1024.0

    print(
        "config: req_num={} rounds={} warmup={} batch_rounds={} "
        "hit_count={} miss_count={} hit_ratio={:.6f} seed={}".format(
            args.req_num,
            args.rounds,
            args.warmup,
            batch_rounds,
            args.hit_count,
            miss_count,
            args.hit_count / QUERY_COUNT,
            args.seed,
        )
    )
    print(f"module: {module_path}")
    print(
        "preload estimate: {:.2f} MiB/state, at most {:.2f} MiB of "
        "input/state/workspace tensors".format(
            state_mib,
            state_mib * (batch_rounds + args.warmup),
        )
    )

    if not args.no_verify:
        verify_one_state(
            np,
            torch,
            module,
            args.req_num,
            args.hit_count,
            args.seed,
        )
        print("verify: PASS")

    completed = 0
    total_wall_ms = 0.0
    total_event_ms = 0.0
    next_case_id = 1
    warmup_remaining = args.warmup
    while completed < args.rounds:
        current_rounds = min(batch_rounds, args.rounds - completed)
        state_count = warmup_remaining + current_rounds
        states = preload_states(
            np,
            torch,
            module,
            args.req_num,
            args.hit_count,
            state_count,
            args.seed,
            next_case_id,
        )
        next_case_id += state_count

        if warmup_remaining:
            warmup_states(
                torch,
                module,
                states[:warmup_remaining],
                args.req_num,
            )
        wall_ms, event_ms = run_states(
            torch,
            module,
            states[warmup_remaining:],
            args.req_num,
        )
        warmup_remaining = 0
        completed += current_rounds
        total_wall_ms += wall_ms
        total_event_ms += event_ms
        del states
        gc.collect()

    launches = args.rounds
    requests = launches * args.req_num
    queries = requests * QUERY_COUNT
    misses = requests * miss_count
    wall_avg_us = total_wall_ms * 1000.0 / launches
    event_avg_us = total_event_ms * 1000.0 / launches
    summary = {
        "module": str(module_path),
        "device": args.device,
        "req_num": args.req_num,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "batch_rounds": batch_rounds,
        "query_count_per_request": QUERY_COUNT,
        "hit_count_per_request": args.hit_count,
        "miss_count_per_request": miss_count,
        "hit_ratio": args.hit_count / QUERY_COUNT,
        "seed": args.seed,
        "random_unique_miss_tokens": True,
        "randomized_miss_positions": True,
        "wall_avg_us_per_launch": wall_avg_us,
        "event_avg_us_per_launch": event_avg_us,
        "event_avg_us_per_request": total_event_ms * 1000.0 / requests,
        "event_ns_per_query": total_event_ms * 1_000_000.0 / queries,
        "event_ns_per_miss": (
            total_event_ms * 1_000_000.0 / misses if misses else None
        ),
    }

    print(
        "summary: wall_avg_us/launch={:.3f} "
        "event_avg_us/launch={:.3f} event_avg_us/request={:.3f} "
        "event_ns/query={:.3f}".format(
            summary["wall_avg_us_per_launch"],
            summary["event_avg_us_per_launch"],
            summary["event_avg_us_per_request"],
            summary["event_ns_per_query"],
        )
    )
    if summary["event_ns_per_miss"] is None:
        print("summary: event_ns/miss=n/a (all-hit workload)")
    else:
        print(
            "summary: event_ns/miss={:.3f}".format(
                summary["event_ns_per_miss"]
            )
        )
    if args.json_output is not None:
        write_json(args.json_output, summary)
        print(f"json: {args.json_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
