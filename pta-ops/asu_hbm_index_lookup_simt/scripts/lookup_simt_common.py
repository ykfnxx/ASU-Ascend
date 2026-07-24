from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PKG_DIR))

from python.lookup_lru_reference import (  # noqa: E402
    LookupState,
    lookup_allocate_evict,
)


@dataclass
class NpuLookupState:
    token_to_slot: Any
    slot_to_token: Any
    lru_slots: Any
    query_token_ids: Any
    workspace: Any


def require_runtime(device_id: int):
    try:
        import numpy as np
        import torch
        import torch_npu
    except Exception as exc:
        raise RuntimeError(
            "runtime requires numpy, torch, torch_npu, and an Ascend 950"
        ) from exc
    torch.npu.set_device(device_id)
    torch.set_grad_enabled(False)
    return np, torch, torch_npu


def find_module_path(build_dir: Path) -> Path:
    candidates = sorted(
        build_dir.expanduser().resolve().rglob("asu_hbm_index_lookup_simt*.so")
    )
    if not candidates:
        raise FileNotFoundError(
            f"could not find asu_hbm_index_lookup_simt*.so under {build_dir}; "
            "pass --module-path explicitly"
        )
    return candidates[0]


def load_extension(module_path: Path | None, build_dir: Path) -> tuple[ModuleType, Path]:
    if module_path is None:
        module_path = find_module_path(build_dir)
    module_path = module_path.expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"extension module does not exist: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "asu_hbm_index_lookup_simt", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module_path


def to_npu_state(torch: Any, module: ModuleType, case: Any) -> NpuLookupState:
    req_num = case.token_to_slot.shape[0]
    return NpuLookupState(
        token_to_slot=torch.from_numpy(case.token_to_slot).to("npu"),
        slot_to_token=torch.from_numpy(case.slot_to_token).to("npu"),
        lru_slots=torch.from_numpy(case.lru_slots).to("npu"),
        query_token_ids=torch.from_numpy(case.query_token_ids).to("npu"),
        workspace=torch.empty(
            module.workspace_size(req_num),
            dtype=torch.int32,
            device="npu",
        ),
    )


def call_lookup(module: ModuleType, state: NpuLookupState, req_num: int):
    return module.asu_hbm_index_lookup_simt(
        state.token_to_slot,
        state.slot_to_token,
        state.lru_slots,
        state.query_token_ids,
        req_num,
        state.workspace,
    )


def expected_result(np: Any, case: Any):
    expected_token_to_slot = case.token_to_slot.copy()
    expected_slot_to_token = case.slot_to_token.copy()
    expected_lru_slots = case.lru_slots.copy()
    expected_slot_ids = np.empty_like(case.query_token_ids)
    expected_miss_mask = np.empty(case.query_token_ids.shape, dtype=np.bool_)

    for req_id in range(case.query_token_ids.shape[0]):
        state = LookupState(
            token_to_slot=expected_token_to_slot[req_id],
            slot_to_token=expected_slot_to_token[req_id],
            lru_slots=expected_lru_slots[req_id],
        )
        slot_ids, miss_mask = lookup_allocate_evict(
            case.query_token_ids[req_id], state
        )
        expected_slot_ids[req_id] = slot_ids
        expected_miss_mask[req_id] = miss_mask

    return (
        expected_token_to_slot,
        expected_slot_to_token,
        expected_lru_slots,
        expected_slot_ids,
        expected_miss_mask,
    )


def assert_runtime_result(
    np: Any,
    state: NpuLookupState,
    outputs: Any,
    expected: Any,
) -> None:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise AssertionError("operator must return (slot_ids, miss_mask)")
    slot_ids, miss_mask = outputs
    np.testing.assert_array_equal(
        state.token_to_slot.cpu().numpy(), expected[0]
    )
    np.testing.assert_array_equal(
        state.slot_to_token.cpu().numpy(), expected[1]
    )
    np.testing.assert_array_equal(state.lru_slots.cpu().numpy(), expected[2])
    np.testing.assert_array_equal(slot_ids.cpu().numpy(), expected[3])
    np.testing.assert_array_equal(miss_mask.cpu().numpy(), expected[4])


def estimate_state_bytes(req_num: int) -> int:
    index_size = 128 * 1024
    slot_count = 10 * 1024
    query_count = 2 * 1024
    workspace_int32 = 3 * slot_count + 3 * 256 + 4
    return req_num * (
        (index_size + slot_count + query_count + workspace_int32) * 4
        + slot_count * 2
    )
