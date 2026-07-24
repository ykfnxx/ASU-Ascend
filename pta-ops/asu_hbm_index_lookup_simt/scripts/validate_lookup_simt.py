#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lookup_simt_common import (
    PKG_DIR,
    assert_runtime_result,
    call_lookup,
    expected_result,
    load_kernel,
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
            "Validate Ascend 950 SIMT lookup with an exact hit count and "
            "random unique miss tokens at randomized query positions."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=PKG_DIR / "build")
    parser.add_argument(
        "--library-path",
        type=Path,
        default=None,
        help="optional path to libasu_hbm_index_lookup_simt_kernel.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=2)
    parser.add_argument(
        "--hit-count",
        type=int,
        default=DEFAULT_HIT_COUNT,
        help=f"exact hits per {QUERY_COUNT}-token request (default: {DEFAULT_HIT_COUNT})",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--case-id", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.req_num <= 0:
        raise ValueError("--req-num must be positive")
    if args.device < 0:
        raise ValueError("--device cannot be negative")
    if args.case_id < 0:
        raise ValueError("--case-id cannot be negative")
    validate_hit_count(args.hit_count)


def main() -> None:
    args = parse_args()
    validate_args(args)
    np, torch, _ = require_runtime(args.device)
    kernel = load_kernel(args.library_path, args.build_dir)
    case = make_random_case(
        np,
        args.req_num,
        args.hit_count,
        seed=args.seed,
        case_id=args.case_id,
    )
    expected = expected_result(np, case)
    state = to_npu_state(torch, case)

    outputs = call_lookup(kernel, torch, state, args.req_num)
    torch.npu.synchronize()
    assert_runtime_result(np, state, outputs, expected)

    miss_positions = np.flatnonzero(
        case.query_token_ids[0] >= 10 * 1024
    )
    print(
        "PASS lookup_simt: req_num={} hit_count={} miss_count={} "
        "seed={} case_id={} library={}".format(
            args.req_num,
            case.hit_count,
            case.miss_count,
            args.seed,
            args.case_id,
            kernel.path,
        )
    )
    print(
        "request0 random miss positions sample={}".format(
            miss_positions[:16].tolist()
        )
    )


if __name__ == "__main__":
    main()
