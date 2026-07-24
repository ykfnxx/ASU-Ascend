from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PKG_DIR))

from python.lookup_lru_reference import (  # noqa: E402
    LookupState,
    lookup_allocate_evict,
)
from python.random_workload import QUERY_COUNT, SLOT_COUNT  # noqa: E402


SIMT_THREADS = 256
WORKSPACE_STRIDE = 3 * SLOT_COUNT + 3 * SIMT_THREADS + 4


@dataclass
class NpuLookupState:
    token_to_slot: Any
    slot_to_token: Any
    lru_slots: Any
    query_token_ids: Any
    slot_ids: Any
    miss_mask: Any
    workspace: Any


@dataclass
class LoadedKernel:
    library: Any
    function: Any
    path: Path


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


def find_library_path(build_dir: Path) -> Path:
    build_root = build_dir.expanduser().resolve()
    expected = build_root / "lib" / "libasu_hbm_index_lookup_simt_kernel.so"
    if expected.is_file():
        return expected
    candidates = sorted(
        build_root.rglob("libasu_hbm_index_lookup_simt_kernel.so")
    )
    if not candidates:
        raise FileNotFoundError(
            "could not find libasu_hbm_index_lookup_simt_kernel.so under "
            f"{build_dir}; pass --library-path explicitly"
        )
    return candidates[0]


def load_kernel(
    library_path: Path | None,
    build_dir: Path,
) -> LoadedKernel:
    if library_path is None:
        library_path = find_library_path(build_dir)
    library_path = library_path.expanduser().resolve()
    if not library_path.is_file():
        raise FileNotFoundError(f"kernel library does not exist: {library_path}")

    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    try:
        function = library.asu_hbm_index_lookup_simt_do
    except AttributeError as exc:
        raise ImportError(
            f"{library_path} does not export asu_hbm_index_lookup_simt_do"
        ) from exc
    function.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_uint32]
    function.restype = None
    return LoadedKernel(library=library, function=function, path=library_path)


def to_npu_state(torch: Any, case: Any) -> NpuLookupState:
    req_num = case.token_to_slot.shape[0]
    query_token_ids = torch.from_numpy(case.query_token_ids).to("npu")
    return NpuLookupState(
        token_to_slot=torch.from_numpy(case.token_to_slot).to("npu"),
        slot_to_token=torch.from_numpy(case.slot_to_token).to("npu"),
        lru_slots=torch.from_numpy(case.lru_slots).to("npu"),
        query_token_ids=query_token_ids,
        slot_ids=torch.empty_like(query_token_ids),
        miss_mask=torch.empty(
            query_token_ids.shape,
            dtype=torch.bool,
            device=query_token_ids.device,
        ),
        workspace=torch.empty(
            req_num * WORKSPACE_STRIDE,
            dtype=torch.int32,
            device="npu",
        ),
    )


def current_stream_ptr(torch: Any) -> int:
    stream_ptr = getattr(torch.npu.current_stream(), "npu_stream", None)
    if stream_ptr is None:
        raise RuntimeError("torch.npu.current_stream() has no npu_stream")
    return int(stream_ptr)


def call_lookup(
    kernel: LoadedKernel,
    torch: Any,
    state: NpuLookupState,
    req_num: int,
):
    kernel.function(
        ctypes.c_void_p(current_stream_ptr(torch)),
        ctypes.c_void_p(state.token_to_slot.data_ptr()),
        ctypes.c_void_p(state.slot_to_token.data_ptr()),
        ctypes.c_void_p(state.lru_slots.data_ptr()),
        ctypes.c_void_p(state.query_token_ids.data_ptr()),
        ctypes.c_void_p(state.slot_ids.data_ptr()),
        ctypes.c_void_p(state.miss_mask.data_ptr()),
        ctypes.c_void_p(state.workspace.data_ptr()),
        ctypes.c_uint32(req_num),
    )
    return state.slot_ids, state.miss_mask


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
    workspace_int32 = WORKSPACE_STRIDE
    return req_num * (
        (index_size + SLOT_COUNT + 2 * QUERY_COUNT + workspace_int32) * 4
        + SLOT_COUNT * 2
        + QUERY_COUNT
    )
