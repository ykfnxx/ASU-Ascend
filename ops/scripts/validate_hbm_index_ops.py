#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from asu_hbm_index_common import (
    DEFAULT_LOOKUP_BUILD_DIR,
    DEFAULT_MAINTAIN_BUILD_DIR,
    FREE_SLOT_COUNT,
    QUERY_COUNT,
    call_lookup,
    call_maintain,
    expected_lookup_allocate,
    expected_maintain,
    format_registered_torch_ops,
    find_lookup_library,
    load_lookup_function,
    make_index_case,
    make_maintain_case,
    require_numpy,
    require_runtime,
    resolve_maintain_callable,
    to_npu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ASU HBM index lookup and maintain ops.")
    parser.add_argument("--target", choices=("lookup", "maintain", "both"), default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=2)
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pattern", choices=("hit", "miss", "mixed"), default="mixed")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_LOOKUP_BUILD_DIR)
    parser.add_argument("--lookup-lib", default=None)
    parser.add_argument("--maintain-build-dir", type=Path, default=DEFAULT_MAINTAIN_BUILD_DIR)
    parser.add_argument("--maintain-lib", default=None, help="Optional direct AICPU library path.")
    parser.add_argument(
        "--evict-ratio",
        type=float,
        default=None,
        help="Optional fraction of the 2K free pool to refill in maintain tests, e.g. 0.25 means 512 slots.",
    )
    parser.add_argument(
        "--maintain-call",
        default=None,
        help="Optional module:function wrapper for the real AICPU op. "
        "Signature: fn(index, slot_to_index, free_slots, free_head, last_query_slots, req_num, seed).",
    )
    parser.add_argument(
        "--maintain-op",
        default=None,
        help="Optional torch.ops name, for example _C_ascend.asu_hbm_index_maintain.",
    )
    parser.add_argument(
        "--op-plugin-lib",
        default=None,
        help="Optional comma-separated torch op plugin libraries to load before resolving --maintain-op.",
    )
    parser.add_argument(
        "--strict-maintain",
        action="store_true",
        help="Fail if no direct AICPU library, torch op, or Python wrapper resolves.",
    )
    parser.add_argument(
        "--list-torch-ops",
        action="store_true",
        help="List registered torch ops matching ASU/index/maintain keywords and exit.",
    )
    return parser.parse_args()


def make_npu_tensors(torch, case, slot_out=None) -> Dict[str, object]:
    tensors = {
        "index": to_npu(torch, case.index),
        "slot_to_index": to_npu(torch, case.slot_to_index),
        "free_slots": to_npu(torch, case.free_slots),
        "free_head": to_npu(torch, case.free_head),
        "query_index": to_npu(torch, case.query_index),
        "slot_out": torch.empty((case.index.shape[0], QUERY_COUNT), dtype=torch.int32).npu(),
    }
    if slot_out is not None:
        tensors["last_query_slots"] = to_npu(torch, slot_out)
    return tensors


def make_maintain_npu_tensors(torch, index, slot_to_index, free_slots, free_head, last_query_slots) -> Dict[str, object]:
    return {
        "index": to_npu(torch, index),
        "slot_to_index": to_npu(torch, slot_to_index),
        "free_slots": to_npu(torch, free_slots),
        "free_head": to_npu(torch, free_head),
        "last_query_slots": to_npu(torch, last_query_slots),
    }


def assert_array_equal(actual, expected) -> None:
    np = require_numpy()
    np.testing.assert_array_equal(actual, expected)


def validate_lookup(args: argparse.Namespace, torch, case) -> Tuple[object, object, object, object]:
    expected = expected_lookup_allocate(case)
    lookup_lib = find_lookup_library(args.build_dir, args.lookup_lib)
    lookup_function = load_lookup_function(lookup_lib)
    tensors = make_npu_tensors(torch, case)

    call_lookup(lookup_function, torch, tensors, args.block_dim, args.req_num)
    torch.npu.synchronize()

    assert_array_equal(tensors["slot_out"].cpu().numpy(), expected.slot_out)
    assert_array_equal(tensors["index"].cpu().numpy(), expected.index)
    assert_array_equal(tensors["slot_to_index"].cpu().numpy(), expected.slot_to_index)
    assert_array_equal(tensors["free_head"].cpu().numpy(), expected.free_head)

    unique_misses = int(expected.free_head.sum())
    print(f"PASS lookup: req_num={args.req_num} pattern={args.pattern} unique_misses={unique_misses}")
    return expected.slot_out, expected.index, expected.slot_to_index, expected.free_head


def validate_maintain(args: argparse.Namespace, torch, case, lookup_state) -> None:
    if args.evict_ratio is None:
        slot_out, index_after_lookup, slot_to_index_after_lookup, free_head_after_lookup = lookup_state
        maintain_index = index_after_lookup
        maintain_slot_to_index = slot_to_index_after_lookup
        maintain_free_slots = case.free_slots
        maintain_free_head = free_head_after_lookup
        maintain_last_query_slots = slot_out
        evict_slots = int(free_head_after_lookup.sum())
        maintain_label = f"lookup_pattern={args.pattern}"
    else:
        maintain_case = make_maintain_case(args.req_num, args.evict_ratio)
        maintain_index = maintain_case.index
        maintain_slot_to_index = maintain_case.slot_to_index
        maintain_free_slots = maintain_case.free_slots
        maintain_free_head = maintain_case.free_head
        maintain_last_query_slots = maintain_case.last_query_slots
        evict_slots = int(maintain_case.free_head.sum())
        maintain_label = f"evict_ratio={args.evict_ratio}"

    expected = expected_maintain(
        maintain_index,
        maintain_slot_to_index,
        maintain_free_slots,
        maintain_free_head,
        maintain_last_query_slots,
        args.seed,
    )

    maintain_function = resolve_maintain_callable(
        torch,
        args.maintain_call,
        args.maintain_op,
        args.op_plugin_lib,
        args.maintain_lib,
        args.maintain_build_dir,
    )
    if maintain_function is None:
        if args.strict_maintain:
            raise RuntimeError(
                "No maintain op resolved. Build maintain_aicpu or pass --maintain-lib for direct AICPU .so. "
                "For framework-registered ops, pass --maintain-op and optionally --op-plugin-lib.\n"
                + format_registered_torch_ops(torch, ("asu", "hbm", "index", "maintain"))
            )
        print("SKIP maintain actual op: pass --maintain-op or --maintain-call after AICPU op integration")
        print(f"PASS maintain reference: {maintain_label} would refill {evict_slots} free slots")
        return

    tensors = make_maintain_npu_tensors(
        torch,
        maintain_index,
        maintain_slot_to_index,
        maintain_free_slots,
        maintain_free_head,
        maintain_last_query_slots,
    )

    call_maintain(maintain_function, torch, tensors, args.block_dim, args.req_num, args.seed)
    torch.npu.synchronize()

    assert_array_equal(tensors["index"].cpu().numpy(), expected[0])
    assert_array_equal(tensors["slot_to_index"].cpu().numpy(), expected[1])
    assert_array_equal(tensors["free_slots"].cpu().numpy(), expected[2])
    assert_array_equal(tensors["free_head"].cpu().numpy(), expected[3])
    print(f"PASS maintain actual op: req_num={args.req_num} {maintain_label} evict_slots={evict_slots}/{args.req_num * FREE_SLOT_COUNT}")


def main() -> None:
    args = parse_args()
    torch = require_runtime(args.device)
    if args.list_torch_ops:
        print(format_registered_torch_ops(torch, ("asu", "hbm", "index", "maintain")))
        return

    case = make_index_case(args.req_num, args.pattern)

    if args.target in ("lookup", "both"):
        lookup_state = validate_lookup(args, torch, case)
    else:
        expected = expected_lookup_allocate(case)
        lookup_state = (expected.slot_out, expected.index, expected.slot_to_index, expected.free_head)

    if args.target in ("maintain", "both"):
        validate_maintain(args, torch, case, lookup_state)


if __name__ == "__main__":
    main()
