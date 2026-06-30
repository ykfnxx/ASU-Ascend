#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
REPO_ROOT = PKG_DIR.parents[1]
OPS_SCRIPTS_DIR = REPO_ROOT / "ops" / "scripts"
sys.path.insert(0, str(OPS_SCRIPTS_DIR))

from asu_hbm_index_common import (  # noqa: E402
    QUERY_COUNT,
    expected_lookup_allocate,
    make_index_case,
    require_numpy,
    require_runtime,
    to_npu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Ascend 950 SIMT PTA ASU HBM index lookup.")
    parser.add_argument("--build-dir", type=Path, default=PKG_DIR / "build")
    parser.add_argument("--module-path", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=2)
    parser.add_argument("--pattern", choices=("hit", "miss", "mixed"), default="mixed")
    return parser.parse_args()


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


def make_npu_tensors(torch, case):
    return {
        "index": to_npu(torch, case.index),
        "slot_to_index": to_npu(torch, case.slot_to_index),
        "free_slots": to_npu(torch, case.free_slots),
        "free_head": to_npu(torch, case.free_head),
        "query_index": to_npu(torch, case.query_index),
    }


def main() -> None:
    args = parse_args()
    torch = require_runtime(args.device)
    module = load_extension(args.module_path, args.build_dir)
    case = make_index_case(args.req_num, args.pattern)
    expected = expected_lookup_allocate(case)
    tensors = make_npu_tensors(torch, case)

    slot_out = module.asu_hbm_index_lookup_simt(
        tensors["index"],
        tensors["slot_to_index"],
        tensors["free_slots"],
        tensors["free_head"],
        tensors["query_index"],
        args.req_num,
    )
    torch.npu.synchronize()

    np = require_numpy()
    np.testing.assert_array_equal(slot_out.cpu().numpy().reshape(args.req_num, QUERY_COUNT), expected.slot_out)
    np.testing.assert_array_equal(tensors["index"].cpu().numpy(), expected.index)
    np.testing.assert_array_equal(tensors["slot_to_index"].cpu().numpy(), expected.slot_to_index)
    np.testing.assert_array_equal(tensors["free_head"].cpu().numpy(), expected.free_head)

    print(
        "PASS lookup_simt: req_num={} pattern={} unique_misses={}".format(
            args.req_num, args.pattern, int(expected.free_head.sum())
        )
    )


if __name__ == "__main__":
    main()
