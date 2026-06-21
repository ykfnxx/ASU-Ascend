#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from asu_hbm_index_common import (
    DEFAULT_LOOKUP_BUILD_DIR,
    DEFAULT_MAINTAIN_BUILD_DIR,
    QUERY_COUNT,
    call_lookup,
    call_maintain,
    expected_lookup_allocate,
    format_registered_torch_ops,
    find_lookup_library,
    load_lookup_function,
    make_chained_maintain_case,
    make_index_case,
    make_maintain_case,
    require_runtime,
    resolve_maintain_callable,
    to_npu,
)


def parse_csv_ints(text: str) -> List[int]:
    return [int(item) for item in text.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ASU HBM index lookup and maintain ops.")
    parser.add_argument("--target", choices=("lookup", "maintain", "both"), default="lookup")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", default="1,2,4,8")
    parser.add_argument("--block-dim", default="1,2,4,8")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--chain-iters",
        type=int,
        default=1,
        help="For maintain, expand req_num by this factor and time one large batch; --iters is ignored when > 1.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pattern", choices=("hit", "miss", "mixed"), default="hit")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_LOOKUP_BUILD_DIR)
    parser.add_argument("--lookup-lib", default=None)
    parser.add_argument("--maintain-build-dir", type=Path, default=DEFAULT_MAINTAIN_BUILD_DIR)
    parser.add_argument("--maintain-lib", default=None, help="Optional direct AICPU library path.")
    parser.add_argument(
        "--evict-ratio",
        type=float,
        default=None,
        help="Optional fraction of the 2K free pool to refill in maintain benchmarks, e.g. 0.25 means 512 slots.",
    )
    parser.add_argument("--maintain-call", default=None, help="module:function wrapper for the real AICPU op")
    parser.add_argument(
        "--maintain-op",
        default=None,
        help="torch.ops name for the real AICPU op, for example _C_ascend.asu_hbm_index_maintain",
    )
    parser.add_argument(
        "--op-plugin-lib",
        default=None,
        help="Optional comma-separated torch op plugin libraries to load before resolving --maintain-op.",
    )
    parser.add_argument(
        "--reset-each-iter",
        action="store_true",
        help="Restore lookup input state before every timed iteration; useful for miss/mixed allocation timing.",
    )
    return parser.parse_args()


def make_lookup_tensors(torch, case):
    return {
        "index": to_npu(torch, case.index),
        "slot_to_index": to_npu(torch, case.slot_to_index),
        "free_slots": to_npu(torch, case.free_slots),
        "free_head": to_npu(torch, case.free_head),
        "query_index": to_npu(torch, case.query_index),
        "slot_out": torch.empty((case.index.shape[0], QUERY_COUNT), dtype=torch.int32).npu(),
    }


def snapshot_tensors(tensors, names: Tuple[str, ...]) -> Dict[str, object]:
    return {name: tensors[name].clone() for name in names}


def restore_tensors(tensors, snapshot: Dict[str, object]) -> None:
    for name, value in snapshot.items():
        tensors[name].copy_(value)


def time_npu_iterations(torch, args, run_once, restore_once=None) -> float:
    for _ in range(args.warmup):
        if restore_once is not None:
            restore_once()
            torch.npu.synchronize()
        run_once()
    torch.npu.synchronize()

    if restore_once is None:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run_once()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end)

    total_ms = 0.0
    for _ in range(args.iters):
        restore_once()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.npu.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms


def bench_lookup(torch, lookup_function, args, req_num: int, block_dim: int) -> None:
    case = make_index_case(req_num, args.pattern)
    tensors = make_lookup_tensors(torch, case)
    snapshot = snapshot_tensors(tensors, ("index", "slot_to_index", "free_slots", "free_head", "query_index"))

    def run_once() -> None:
        call_lookup(lookup_function, torch, tensors, block_dim, req_num)

    restore_once = (lambda: restore_tensors(tensors, snapshot)) if args.reset_each_iter else None
    total_ms = time_npu_iterations(torch, args, run_once, restore_once)
    queries = args.iters * req_num * QUERY_COUNT
    qps = queries / (total_ms / 1000.0)
    print(f"lookup\t{req_num}\t{block_dim}\t{args.pattern}\t{args.iters}\t{total_ms / args.iters:.6f}\t{qps:.3f}")


def bench_maintain(torch, maintain_function, args, req_num: int, block_dim: int) -> None:
    tensors, maintained_slots_per_iter, pattern = make_maintain_benchmark_tensors(torch, args, req_num)
    snapshot = snapshot_tensors(tensors, ("index", "slot_to_index", "free_slots", "free_head", "last_query_slots"))

    def restore_once() -> None:
        restore_tensors(tensors, snapshot)

    iteration = 0

    def run_once() -> None:
        nonlocal iteration
        call_maintain(maintain_function, torch, tensors, block_dim, req_num, args.seed + iteration)
        iteration += 1

    total_ms = time_npu_iterations(torch, args, run_once, restore_once)
    maintained_slots = args.iters * maintained_slots_per_iter
    slots_per_second = maintained_slots / (total_ms / 1000.0) if maintained_slots else 0.0
    print(f"maintain\t{req_num}\t{block_dim}\t{pattern}\t{args.iters}\t{total_ms / args.iters:.6f}\t{slots_per_second:.3f}")


def make_maintain_benchmark_tensors(torch, args, req_num: int):
    if args.evict_ratio is None:
        case = make_index_case(req_num, "mixed")
        lookup_expected = expected_lookup_allocate(case)
        return {
            "index": to_npu(torch, lookup_expected.index),
            "slot_to_index": to_npu(torch, lookup_expected.slot_to_index),
            "free_slots": to_npu(torch, case.free_slots),
            "free_head": to_npu(torch, lookup_expected.free_head),
            "last_query_slots": to_npu(torch, lookup_expected.slot_out),
        }, int(lookup_expected.free_head.sum()), "mixed"

    maintain_case = make_maintain_case(req_num, args.evict_ratio)
    return {
        "index": to_npu(torch, maintain_case.index),
        "slot_to_index": to_npu(torch, maintain_case.slot_to_index),
        "free_slots": to_npu(torch, maintain_case.free_slots),
        "free_head": to_npu(torch, maintain_case.free_head),
        "last_query_slots": to_npu(torch, maintain_case.last_query_slots),
    }, int(maintain_case.free_head.sum()), f"evict={args.evict_ratio}"


def bench_maintain_chained(torch, maintain_function, args, req_num: int, block_dim: int) -> None:
    effective_req_num = req_num * args.chain_iters

    def make_tensors():
        if args.evict_ratio is None:
            return make_maintain_benchmark_tensors(torch, args, effective_req_num)

        maintain_case = make_chained_maintain_case(req_num, args.evict_ratio, args.chain_iters)
        return {
            "index": to_npu(torch, maintain_case.index),
            "slot_to_index": to_npu(torch, maintain_case.slot_to_index),
            "free_slots": to_npu(torch, maintain_case.free_slots),
            "free_head": to_npu(torch, maintain_case.free_head),
            "last_query_slots": to_npu(torch, maintain_case.last_query_slots),
        }, int(maintain_case.free_head.sum()), f"evict={args.evict_ratio}"

    for warmup_id in range(args.warmup):
        tensors, _, _ = make_tensors()
        call_maintain(maintain_function, torch, tensors, block_dim, effective_req_num, args.seed + warmup_id)
        torch.npu.synchronize()

    tensors, maintained_slots, pattern = make_tensors()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    call_maintain(maintain_function, torch, tensors, block_dim, effective_req_num, args.seed)
    end.record()
    torch.npu.synchronize()
    total_ms = start.elapsed_time(end)

    pattern = f"{pattern},chain={args.chain_iters}"
    slots_per_second = maintained_slots / (total_ms / 1000.0) if maintained_slots else 0.0
    print(f"maintain\t{req_num}\t{block_dim}\t{pattern}\t{args.chain_iters}\t{total_ms / args.chain_iters:.6f}\t{slots_per_second:.3f}")


def main() -> None:
    args = parse_args()
    torch = require_runtime(args.device)
    lookup_function = None
    maintain_function = None

    if args.target in ("lookup", "both"):
        lookup_lib = find_lookup_library(args.build_dir, args.lookup_lib)
        lookup_function = load_lookup_function(lookup_lib)
    if args.target in ("maintain", "both"):
        maintain_function = resolve_maintain_callable(
            torch,
            args.maintain_call,
            args.maintain_op,
            args.op_plugin_lib,
            args.maintain_lib,
            args.maintain_build_dir,
        )
        if maintain_function is None:
            raise RuntimeError(
                "No maintain op resolved. Build maintain_aicpu or pass --maintain-lib for direct AICPU .so. "
                "For framework-registered ops, pass --maintain-op and optionally --op-plugin-lib.\n"
                + format_registered_torch_ops(torch, ("asu", "hbm", "index", "maintain"))
            )

    print("target\treq_num\tblock_dim\tpattern\titers\tdevice_ms_per_iter\tthroughput_per_second")
    for req_num in parse_csv_ints(args.req_num):
        if args.target in ("lookup", "both"):
            for block_dim in parse_csv_ints(args.block_dim):
                bench_lookup(torch, lookup_function, args, req_num, block_dim)
        if args.target in ("maintain", "both"):
            for block_dim in parse_csv_ints(args.block_dim):
                if args.chain_iters > 1:
                    bench_maintain_chained(torch, maintain_function, args, req_num, block_dim)
                else:
                    bench_maintain(torch, maintain_function, args, req_num, block_dim)


if __name__ == "__main__":
    main()
