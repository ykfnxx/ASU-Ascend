from __future__ import annotations

import ctypes
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
RESIDENT_SLOT_COUNT = 8 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
NOT_FOUND = -1
SCRIPT_DIR = Path(__file__).resolve().parent
OPS_DIR = SCRIPT_DIR.parent
DEFAULT_LOOKUP_BUILD_DIR = OPS_DIR / "build" / "lookup_aiv"
DEFAULT_MAINTAIN_BUILD_DIR = OPS_DIR / "build" / "maintain_aicpu"
DEFAULT_MAINTAIN_OP_CANDIDATES = (
    "_C_ascend.asu_hbm_index_maintain",
    "_C_ascend.AsuHbmIndexMaintain",
    "custom_ops.asu_hbm_index_maintain",
    "custom_ops.AsuHbmIndexMaintain",
    "ascend.asu_hbm_index_maintain",
    "ascend.AsuHbmIndexMaintain",
)


@dataclass
class IndexCase:
    index: np.ndarray
    slot_to_index: np.ndarray
    free_slots: np.ndarray
    free_head: np.ndarray
    query_index: np.ndarray


@dataclass
class LookupExpected:
    slot_out: np.ndarray
    index: np.ndarray
    slot_to_index: np.ndarray
    free_head: np.ndarray


@dataclass
class MaintainCase:
    index: np.ndarray
    slot_to_index: np.ndarray
    free_slots: np.ndarray
    free_head: np.ndarray
    last_query_slots: np.ndarray
    evict_slots: int


@dataclass
class MaintainRunner:
    kind: str
    function: Callable


def require_numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for ASU HBM index test data generation") from exc
    return np


def hash32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def make_index_case(req_num: int, pattern: str) -> IndexCase:
    np = require_numpy()
    index = np.full((req_num, INDEX_SIZE), NOT_FOUND, dtype=np.int32)
    slot_to_index = np.full((req_num, SLOT_COUNT), NOT_FOUND, dtype=np.int32)
    free_slots = np.empty((req_num, FREE_SLOT_COUNT), dtype=np.int32)
    free_head = np.zeros((req_num,), dtype=np.int32)
    query_index = np.empty((req_num, QUERY_COUNT), dtype=np.int32)

    resident_keys = np.arange(RESIDENT_SLOT_COUNT, dtype=np.int32)
    free_slot_ids = np.arange(RESIDENT_SLOT_COUNT, SLOT_COUNT, dtype=np.int32)

    for req_id in range(req_num):
        index[req_id, resident_keys] = resident_keys
        slot_to_index[req_id, resident_keys] = resident_keys
        free_slots[req_id] = free_slot_ids
        query_index[req_id] = make_query(pattern, req_id)

    return IndexCase(index, slot_to_index, free_slots, free_head, query_index)


def make_query(pattern: str, req_id: int) -> np.ndarray:
    np = require_numpy()
    hit_query = np.arange(QUERY_COUNT, dtype=np.int32) % RESIDENT_SLOT_COUNT
    miss_base = 20_000 + req_id * 3_000
    miss_query = (miss_base + np.arange(QUERY_COUNT, dtype=np.int32)) % INDEX_SIZE

    if pattern == "hit":
        return hit_query
    if pattern == "miss":
        return miss_query
    if pattern == "mixed":
        query = hit_query.copy()
        query[QUERY_COUNT // 2 :] = miss_query[: QUERY_COUNT // 2]
        query[QUERY_COUNT // 2 + 1] = query[QUERY_COUNT // 2]
        query[QUERY_COUNT // 2 + 3] = query[QUERY_COUNT // 2 + 2]
        return query
    raise ValueError(f"unsupported query pattern: {pattern}")


def expected_lookup_allocate(case: IndexCase) -> LookupExpected:
    np = require_numpy()
    index = case.index.copy()
    slot_to_index = case.slot_to_index.copy()
    free_head = case.free_head.copy()
    slot_out = np.empty_like(case.query_index)

    for req_id in range(case.query_index.shape[0]):
        head = int(free_head[req_id])
        for query_pos, index_id_np in enumerate(case.query_index[req_id]):
            index_id = int(index_id_np)
            slot = int(index[req_id, index_id])
            if slot == NOT_FOUND:
                slot = int(case.free_slots[req_id, head])
                head += 1
                index[req_id, index_id] = slot
                slot_to_index[req_id, slot] = index_id
            slot_out[req_id, query_pos] = slot
        free_head[req_id] = head

    return LookupExpected(slot_out, index, slot_to_index, free_head)


def evict_slots_from_ratio(evict_ratio: float) -> int:
    if evict_ratio < 0.0 or evict_ratio > 1.0:
        raise ValueError("--evict-ratio must be in [0, 1]")
    return int(round(FREE_SLOT_COUNT * evict_ratio))


def make_maintain_case(req_num: int, evict_ratio: float) -> MaintainCase:
    np = require_numpy()
    evict_slots = evict_slots_from_ratio(evict_ratio)

    index = np.full((req_num, INDEX_SIZE), NOT_FOUND, dtype=np.int32)
    slot_to_index = np.full((req_num, SLOT_COUNT), NOT_FOUND, dtype=np.int32)
    free_slots = np.empty((req_num, FREE_SLOT_COUNT), dtype=np.int32)
    free_head = np.full((req_num,), evict_slots, dtype=np.int32)
    last_query_slots = np.empty((req_num, QUERY_COUNT), dtype=np.int32)

    resident_keys = np.arange(RESIDENT_SLOT_COUNT, dtype=np.int32)
    free_slot_ids = np.arange(RESIDENT_SLOT_COUNT, SLOT_COUNT, dtype=np.int32)
    miss_keys = 20_000 + np.arange(evict_slots, dtype=np.int32)
    allocated_slots = RESIDENT_SLOT_COUNT + np.arange(evict_slots, dtype=np.int32)
    hit_count = QUERY_COUNT - evict_slots

    for req_id in range(req_num):
        index[req_id, resident_keys] = resident_keys
        slot_to_index[req_id, resident_keys] = resident_keys
        free_slots[req_id] = free_slot_ids
        if evict_slots > 0:
            index[req_id, miss_keys] = allocated_slots
            slot_to_index[req_id, allocated_slots] = miss_keys
        if hit_count > 0:
            last_query_slots[req_id, :hit_count] = np.arange(hit_count, dtype=np.int32)
        if evict_slots > 0:
            last_query_slots[req_id, hit_count:] = allocated_slots

    return MaintainCase(index, slot_to_index, free_slots, free_head, last_query_slots, evict_slots)


def make_chained_maintain_case(req_num: int, evict_ratio: float, chain_iters: int) -> MaintainCase:
    if chain_iters < 1:
        raise ValueError("--chain-iters must be >= 1")
    return make_maintain_case(req_num * chain_iters, evict_ratio)


def expected_maintain(
    index: np.ndarray,
    slot_to_index: np.ndarray,
    free_slots: np.ndarray,
    free_head: np.ndarray,
    last_query_slots: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected_index = index.copy()
    expected_slot_to_index = slot_to_index.copy()
    expected_free_slots = free_slots.copy()
    expected_free_head = free_head.copy()

    for req_id in range(index.shape[0]):
        protected_slots = {int(slot) for slot in last_query_slots[req_id]}
        head = int(expected_free_head[req_id])
        start = hash32(seed ^ req_id) % SLOT_COUNT
        offset = 0
        while head > 0:
            slot = (start + offset) % SLOT_COUNT
            index_id = int(expected_slot_to_index[req_id, slot])
            if index_id != NOT_FOUND and slot not in protected_slots:
                expected_slot_to_index[req_id, slot] = NOT_FOUND
                expected_index[req_id, index_id] = NOT_FOUND
                head -= 1
                expected_free_slots[req_id, head] = slot
            offset += 1
        expected_free_head[req_id] = head

    return expected_index, expected_slot_to_index, expected_free_slots, expected_free_head


def require_runtime(device: int):
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise RuntimeError("torch and torch_npu are required on the Ascend runtime host") from exc

    torch.npu.set_device(device)
    return torch


def to_npu(torch, array: np.ndarray):
    np = require_numpy()
    return torch.from_numpy(np.ascontiguousarray(array)).npu()


def current_stream_ptr(torch) -> int:
    stream = torch.npu.current_stream()
    stream_ptr = getattr(stream, "npu_stream", None)
    if stream_ptr is None:
        raise RuntimeError("torch.npu.current_stream() has no npu_stream attribute")
    return int(stream_ptr)


def require_npu_tensors(tensors: Dict[str, object], names: Iterable[str]) -> None:
    for name in names:
        tensor = tensors[name]
        device = getattr(tensor, "device", None)
        device_type = getattr(device, "type", None)
        if device_type != "npu":
            raise RuntimeError("{} must be an NPU tensor; got device {}".format(name, device))


def find_lookup_library(build_dir: Path, explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"lookup library does not exist: {path}")
        return path

    search_root = build_dir.expanduser().resolve()
    candidates = sorted(search_root.rglob("libasu_hbm_index_lookup_aiv*.so"))
    if not candidates:
        raise FileNotFoundError(
            f"could not find libasu_hbm_index_lookup_aiv*.so under {search_root}; "
            "pass --lookup-lib explicitly"
        )
    return candidates[0]


def find_maintain_library(build_dir: Path, explicit_path: Optional[str] = None) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("maintain library does not exist: {}".format(path))
        return path

    search_root = build_dir.expanduser().resolve()
    if not search_root.exists():
        return None
    candidates = sorted(search_root.rglob("libasu_hbm_index_maintain_aicpu*.so"))
    if not candidates:
        return None
    return candidates[0]


def load_lookup_function(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    function = library.asu_hbm_index_lookup_do
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    function.restype = None
    return function


def load_maintain_function(library_path: Path):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    function = library.asu_hbm_index_maintain_do
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    function.restype = None
    return function


def call_lookup(function, torch, tensors: Dict[str, object], block_dim: int, req_num: int) -> None:
    require_npu_tensors(tensors, ("index", "slot_to_index", "free_slots", "free_head", "query_index", "slot_out"))
    function(
        ctypes.c_uint32(block_dim),
        ctypes.c_void_p(current_stream_ptr(torch)),
        ctypes.c_void_p(tensors["index"].data_ptr()),
        ctypes.c_void_p(tensors["slot_to_index"].data_ptr()),
        ctypes.c_void_p(tensors["free_slots"].data_ptr()),
        ctypes.c_void_p(tensors["free_head"].data_ptr()),
        ctypes.c_void_p(tensors["query_index"].data_ptr()),
        ctypes.c_void_p(tensors["slot_out"].data_ptr()),
        ctypes.c_uint32(req_num),
    )


def call_maintain_direct(function, torch, tensors: Dict[str, object], block_dim: int, req_num: int, seed: int) -> None:
    require_npu_tensors(tensors, ("index", "slot_to_index", "free_slots", "free_head", "last_query_slots"))
    function(
        ctypes.c_uint32(block_dim),
        ctypes.c_void_p(current_stream_ptr(torch)),
        ctypes.c_void_p(tensors["index"].data_ptr()),
        ctypes.c_void_p(tensors["slot_to_index"].data_ptr()),
        ctypes.c_void_p(tensors["free_slots"].data_ptr()),
        ctypes.c_void_p(tensors["free_head"].data_ptr()),
        ctypes.c_void_p(tensors["last_query_slots"].data_ptr()),
        ctypes.c_uint32(req_num),
        ctypes.c_uint32(seed),
    )


def resolve_python_callable(spec: Optional[str]) -> Optional[Callable]:
    if not spec:
        return None
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("--maintain-call must use module:function format")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


def load_op_plugin(torch, library_path: Optional[str]) -> None:
    if not library_path:
        return
    for item in library_path.split(","):
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("op plugin library does not exist: {}".format(path))
        torch.ops.load_library(str(path))


def normalize_torch_op_name(spec: str) -> str:
    if "::" in spec:
        namespace, op_name = spec.split("::", 1)
        return "{}.{}".format(namespace, op_name)
    return spec


def list_registered_torch_ops(torch, keywords: Iterable[str]) -> List[str]:
    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
    matches = []
    try:
        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return matches

    for schema in schemas:
        name = getattr(schema, "name", "")
        if not name:
            text = str(schema)
            name = text.split("(", 1)[0]
        lowered_name = name.lower()
        if any(keyword in lowered_name for keyword in lowered_keywords):
            matches.append(name)
    return sorted(set(matches))


def format_registered_torch_ops(torch, keywords: Iterable[str]) -> str:
    matches = list_registered_torch_ops(torch, keywords)
    if not matches:
        return "No registered torch ops matched keywords: {}".format(", ".join(keywords))
    return "Registered torch ops matching keywords:\n  {}".format("\n  ".join(matches))


def resolve_torch_op(torch, spec: str) -> Callable:
    spec = normalize_torch_op_name(spec)
    target = torch.ops
    for part in spec.split("."):
        if not part:
            raise ValueError("--maintain-op must use namespace.op format")
        target = getattr(target, part)
    if not callable(target):
        raise TypeError("{} is not callable".format(spec))
    return target


def resolve_maintain_callable(
    torch,
    maintain_call: Optional[str],
    maintain_op: Optional[str],
    op_plugin_lib: Optional[str],
    maintain_lib: Optional[str],
    maintain_build_dir: Path,
) -> Optional[MaintainRunner]:
    load_op_plugin(torch, op_plugin_lib)

    maintain_library = find_maintain_library(maintain_build_dir, maintain_lib)
    if maintain_library is not None:
        return MaintainRunner("direct", load_maintain_function(maintain_library))

    if maintain_call:
        return MaintainRunner("python", resolve_python_callable(maintain_call))
    if maintain_op:
        try:
            return MaintainRunner("python", resolve_torch_op(torch, maintain_op))
        except AttributeError as exc:
            diagnostics = format_registered_torch_ops(torch, ("asu", "hbm", "index", "maintain"))
            raise RuntimeError(
                "Could not resolve --maintain-op '{}'.\n{}\n"
                "This usually means the AICPU op package was not built/installed/loaded into torch.ops. "
                "If you have a plugin .so, pass it with --op-plugin-lib first.".format(maintain_op, diagnostics)
            ) from exc

    for candidate in DEFAULT_MAINTAIN_OP_CANDIDATES:
        try:
            return MaintainRunner("python", resolve_torch_op(torch, candidate))
        except (AttributeError, RuntimeError):
            continue
    return None


def call_maintain_python(function: Callable, tensors: Dict[str, object], req_num: int, seed: int) -> None:
    result = function(
        tensors["index"],
        tensors["slot_to_index"],
        tensors["free_slots"],
        tensors["free_head"],
        tensors["last_query_slots"],
        req_num,
        seed,
    )
    if result is None:
        return
    if len(result) != 4:
        raise RuntimeError("maintain callable must return None or four tensors")
    for name, output in zip(("index", "slot_to_index", "free_slots", "free_head"), result):
        tensors[name].copy_(output)


def call_maintain(runner: MaintainRunner, torch, tensors: Dict[str, object], block_dim: int, req_num: int, seed: int) -> None:
    if runner.kind == "direct":
        call_maintain_direct(runner.function, torch, tensors, block_dim, req_num, seed)
        return
    call_maintain_python(runner.function, tensors, req_num, seed)
