#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
REPO_ROOT = PKG_DIR.parents[1]
OPS_SCRIPTS_DIR = REPO_ROOT / "ops" / "scripts"
sys.path.insert(0, str(OPS_SCRIPTS_DIR))

from asu_hbm_index_common import (  # noqa: E402
    FREE_SLOT_COUNT,
    INDEX_SIZE,
    QUERY_COUNT,
    RESIDENT_SLOT_COUNT,
    SLOT_COUNT,
    expected_lookup_allocate,
    make_index_case,
    require_numpy,
    require_runtime,
    to_npu,
)


@dataclass
class BenchmarkState:
    index: object
    slot_to_index: object
    free_slots: object
    free_head: object
    query_index: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Ascend 950 SIMT PTA lookup with preloaded fresh NPU states."
    )
    parser.add_argument("--build-dir", type=Path, default=PKG_DIR / "build")
    parser.add_argument("--module-path", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-rounds", type=int, default=0)
    parser.add_argument("--miss-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep hit queries before miss queries.")
    parser.add_argument("--no-verify", action="store_true", help="Skip one fresh-state correctness check before timing.")
    args = parser.parse_args()

    if args.req_num <= 0:
        parser.error("--req-num must be positive")
    if args.rounds < 100:
        parser.error("--rounds must be at least 100 for this benchmark")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.batch_rounds < 0:
        parser.error("--batch-rounds must be non-negative")
    if args.miss_ratio < 0.0 or args.miss_ratio > 1.0:
        parser.error("--miss-ratio must be in [0, 1]")
    return args


def find_module_path(build_dir: Path) -> Path:
    candidates = sorted(build_dir.expanduser().resolve().rglob("asu_hbm_index_lookup_simt*.so"))
    if not candidates:
        raise FileNotFoundError(
            f"could not find asu_hbm_index_lookup_simt*.so under {build_dir}; "
            "pass --module-path explicitly"
        )
    return candidates[0]


def load_extension(module_path: Path | None, build_dir: Path) -> ModuleType:
    if module_path is None:
        module_path = find_module_path(build_dir)
    module_path = module_path.expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"extension module does not exist: {module_path}")

    spec = importlib.util.spec_from_file_location("asu_hbm_index_lookup_simt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unique_misses_per_req(miss_ratio: float) -> int:
    miss_count = int(round(QUERY_COUNT * miss_ratio))
    if miss_count > FREE_SLOT_COUNT:
        raise ValueError(
            f"miss_ratio={miss_ratio} creates {miss_count} distinct misses per request, "
            f"but only {FREE_SLOT_COUNT} free slots are available"
        )
    return miss_count


def make_ratio_case(req_num: int, miss_ratio: float, seed: int, case_id: int, shuffle: bool):
    np = require_numpy()
    case = make_index_case(req_num, "hit")
    miss_count = unique_misses_per_req(miss_ratio)
    hit_count = QUERY_COUNT - miss_count
    hit_query = np.arange(hit_count, dtype=np.int32) % RESIDENT_SLOT_COUNT
    miss_pool = np.arange(RESIDENT_SLOT_COUNT, INDEX_SIZE, dtype=np.int32)
    if miss_count > miss_pool.size:
        raise ValueError("miss count is larger than available miss key pool")
    rng = np.random.default_rng(seed + case_id)

    for req_id in range(req_num):
        start = ((case_id * req_num + req_id) * max(miss_count, 1)) % (miss_pool.size - miss_count + 1)
        misses = miss_pool[start : start + miss_count]
        query = np.empty((QUERY_COUNT,), dtype=np.int32)
        query[:hit_count] = hit_query
        if miss_count:
            query[hit_count:] = misses
        if shuffle:
            rng.shuffle(query)
        case.query_index[req_id] = query
    return case


def preload_benchmark_states(torch, req_num: int, miss_ratio: float, count: int, seed: int, first_case_id: int, shuffle: bool):
    states: List[BenchmarkState] = []
    for offset in range(count):
        case = make_ratio_case(req_num, miss_ratio, seed, first_case_id + offset, shuffle)
        states.append(
            BenchmarkState(
                index=to_npu(torch, case.index),
                slot_to_index=to_npu(torch, case.slot_to_index),
                free_slots=to_npu(torch, case.free_slots),
                free_head=to_npu(torch, case.free_head),
                query_index=to_npu(torch, case.query_index),
            )
        )
    torch.npu.synchronize()
    return states


def estimate_state_bytes(req_num: int) -> int:
    int32_bytes = 4
    per_req = INDEX_SIZE + SLOT_COUNT + FREE_SLOT_COUNT + QUERY_COUNT
    return (req_num * per_req + req_num) * int32_bytes


def call_lookup(module: ModuleType, state: BenchmarkState, req_num: int):
    return module.asu_hbm_index_lookup_simt(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        state.query_index,
        req_num,
    )


def verify_one_state(torch, module: ModuleType, req_num: int, miss_ratio: float, seed: int, shuffle: bool) -> None:
    np = require_numpy()
    case = make_ratio_case(req_num, miss_ratio, seed, 0, shuffle)
    expected = expected_lookup_allocate(case)
    state = BenchmarkState(
        index=to_npu(torch, case.index),
        slot_to_index=to_npu(torch, case.slot_to_index),
        free_slots=to_npu(torch, case.free_slots),
        free_head=to_npu(torch, case.free_head),
        query_index=to_npu(torch, case.query_index),
    )
    slot_out = call_lookup(module, state, req_num)
    torch.npu.synchronize()
    np.testing.assert_array_equal(slot_out.cpu().numpy().reshape(req_num, QUERY_COUNT), expected.slot_out)
    np.testing.assert_array_equal(state.index.cpu().numpy(), expected.index)
    np.testing.assert_array_equal(state.slot_to_index.cpu().numpy(), expected.slot_to_index)
    np.testing.assert_array_equal(state.free_head.cpu().numpy(), expected.free_head)


def run_states(torch, module: ModuleType, states: Sequence[BenchmarkState], req_num: int):
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
    event_ms = start_event.elapsed_time(end_event)
    return wall_ms, event_ms, outputs


def main() -> None:
    args = parse_args()
    torch = require_runtime(args.device)
    module = load_extension(args.module_path, args.build_dir)
    shuffle = not args.no_shuffle
    miss_count = unique_misses_per_req(args.miss_ratio)
    batch_rounds = args.batch_rounds or args.rounds
    if batch_rounds > args.rounds:
        batch_rounds = args.rounds

    state_bytes = estimate_state_bytes(args.req_num)
    peak_states = args.warmup + batch_rounds
    print(
        "config: req_num={} rounds={} warmup={} batch_rounds={} miss_ratio={:.4f} "
        "unique_misses_per_req={} shuffle={}".format(
            args.req_num, args.rounds, args.warmup, batch_rounds, args.miss_ratio, miss_count, shuffle
        )
    )
    print(
        "preload estimate: {:.2f} MiB per state, {:.2f} MiB for warmup+one batch".format(
            state_bytes / 1024.0 / 1024.0, state_bytes * peak_states / 1024.0 / 1024.0
        )
    )

    if not args.no_verify:
        verify_one_state(torch, module, args.req_num, args.miss_ratio, args.seed, shuffle)
        print("verify: PASS")

    completed = 0
    total_wall_ms = 0.0
    total_event_ms = 0.0
    next_case_id = 1
    batch_id = 0

    while completed < args.rounds:
        current_rounds = min(batch_rounds, args.rounds - completed)
        total_states = args.warmup + current_rounds
        states = preload_benchmark_states(
            torch,
            args.req_num,
            args.miss_ratio,
            total_states,
            args.seed,
            next_case_id,
            shuffle,
        )
        next_case_id += total_states

        warmup_states = states[: args.warmup]
        timed_states = states[args.warmup :]
        if warmup_states:
            _, _, warmup_outputs = run_states(torch, module, warmup_states, args.req_num)
            del warmup_outputs
            torch.npu.synchronize()

        wall_ms, event_ms, outputs = run_states(torch, module, timed_states, args.req_num)
        del outputs
        torch.npu.synchronize()

        completed += current_rounds
        total_wall_ms += wall_ms
        total_event_ms += event_ms
        batch_id += 1
        print(
            "batch {}: rounds={} wall_ms={:.3f} event_ms={:.3f} wall_avg_us={:.3f} event_avg_us={:.3f}".format(
                batch_id,
                current_rounds,
                wall_ms,
                event_ms,
                wall_ms * 1000.0 / current_rounds,
                event_ms * 1000.0 / current_rounds,
            )
        )

    total_unique_misses = args.rounds * args.req_num * miss_count
    print(
        "summary: rounds={} wall_ms={:.3f} event_ms={:.3f} wall_avg_us={:.3f} event_avg_us={:.3f} "
        "unique_misses={} wall_ns_per_unique_miss={:.3f}".format(
            args.rounds,
            total_wall_ms,
            total_event_ms,
            total_wall_ms * 1000.0 / args.rounds,
            total_event_ms * 1000.0 / args.rounds,
            total_unique_misses,
            total_wall_ms * 1_000_000.0 / total_unique_misses if total_unique_misses else 0.0,
        )
    )


if __name__ == "__main__":
    main()
