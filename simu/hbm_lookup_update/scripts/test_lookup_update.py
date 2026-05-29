import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch_npu  # noqa: F401

# If executed from project root after CMake build.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.environ.get("BUILD_DIR", os.path.join(ROOT, "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import hbm_lookup_update  # noqa: E402

TABLE_SIZE = 2048
NOT_FOUND = -1


def u32(x: int) -> int:
    return x & 0xFFFFFFFF


def hash32(x: int) -> int:
    x = u32(x)
    x ^= x >> 16
    x = u32(x * 0x7FEB352D)
    x ^= x >> 15
    x = u32(x * 0x846CA68B)
    x ^= x >> 16
    return u32(x)


def pick_coprime_a(seed: int, n: int) -> int:
    if n <= 1:
        return 1
    a = hash32(seed) % n
    if a == 0:
        a = 1
    while math.gcd(a, n) != 1:
        a += 1
        if a >= n:
            a = 1
    return a


def update_positions(query_len: int, seed: int, update_percent: int, req_id: int = 0) -> List[int]:
    update_num = (query_len * update_percent) // 100
    if update_num <= 0:
        return []
    req_seed = u32(seed) ^ hash32(req_id)
    a = pick_coprime_a(req_seed ^ 0x9E3779B9, query_len)
    b = hash32(req_seed ^ 0x85EBCA6B) % query_len
    return [((a * t + b) % query_len) for t in range(update_num)]


def first_key_to_idx(table_keys_cpu: np.ndarray) -> Dict[int, int]:
    key_to_idx: Dict[int, int] = {}
    for i, key in enumerate(table_keys_cpu):
        key_to_idx.setdefault(int(key), int(i))
    return key_to_idx


def expected_lookup_update(
    table_keys_cpu: np.ndarray,
    table_states_cpu: np.ndarray,
    query_cpu: np.ndarray,
    new_states_cpu: np.ndarray,
    seed: int,
    update_percent: int,
) -> tuple[np.ndarray, np.ndarray]:
    single_req = table_keys_cpu.ndim == 1
    table_keys_2d = table_keys_cpu.reshape(1, TABLE_SIZE) if single_req else table_keys_cpu
    table_states_2d = table_states_cpu.reshape(1, TABLE_SIZE) if single_req else table_states_cpu
    query_2d = query_cpu.reshape(1, query_cpu.shape[-1]) if single_req else query_cpu
    new_states_2d = new_states_cpu.reshape(1, new_states_cpu.shape[-1]) if single_req else new_states_cpu

    req_num, query_len = query_2d.shape
    expected_out = np.empty((req_num, query_len), dtype=np.int32)
    expected_states = table_states_2d.copy()

    for req_id in range(req_num):
        key_to_idx = first_key_to_idx(table_keys_2d[req_id])
        for i, qk in enumerate(query_2d[req_id]):
            idx = key_to_idx.get(int(qk))
            expected_out[req_id, i] = NOT_FOUND if idx is None else table_states_2d[req_id, idx]

        for pos in update_positions(query_len, seed, update_percent, req_id):
            qk = int(query_2d[req_id, pos])
            idx = key_to_idx.get(qk)
            if idx is not None:
                expected_states[req_id, idx] = new_states_2d[req_id, pos]

    if single_req:
        return expected_out.reshape(query_cpu.shape), expected_states.reshape(table_states_cpu.shape)
    return expected_out, expected_states


def run_single_req_case(rng: np.random.Generator):
    # table_keys is deliberately shuffled to prove this is comparison-based lookup,
    # not table_states[query_key] dense indexing.
    table_keys_cpu = rng.permutation(np.arange(TABLE_SIZE, dtype=np.int32)).astype(np.int32)
    table_states_cpu = (table_keys_cpu.astype(np.int64) * 10 + 7).astype(np.int32)

    query_len = 512
    query_cpu = rng.choice(np.arange(TABLE_SIZE, dtype=np.int32), size=query_len, replace=True).astype(np.int32)
    # Add several misses.
    query_cpu[::97] = (3000 + np.arange(len(query_cpu[::97]))).astype(np.int32)
    new_states_cpu = (100000 + np.arange(query_len, dtype=np.int32)).astype(np.int32)

    seed = 42
    update_percent = 5
    expected_out, expected_states = expected_lookup_update(
        table_keys_cpu, table_states_cpu, query_cpu, new_states_cpu, seed, update_percent)

    table_keys = torch.from_numpy(table_keys_cpu).npu()
    table_states = torch.from_numpy(table_states_cpu.copy()).npu()
    query_keys = torch.from_numpy(query_cpu).npu()
    new_states = torch.from_numpy(new_states_cpu).npu()

    out = hbm_lookup_update.lookup_random_update(
        table_keys,
        table_states,
        query_keys,
        new_states,
        seed=seed,
        update_percent=update_percent,
        block_dim=8,
        not_found=NOT_FOUND,
        do_update=True,
    )
    torch.npu.synchronize()

    out_cpu = out.cpu().numpy()
    states_after_cpu = table_states.cpu().numpy()

    np.testing.assert_array_equal(out_cpu, expected_out)
    np.testing.assert_array_equal(states_after_cpu, expected_states)

    print("PASS: single-req lookup output matches pre-update states")
    print(f"PASS: single-req table_states updated at {len(update_positions(query_len, seed, update_percent))} query positions")


def run_multi_req_case(rng: np.random.Generator):
    req_num = 4
    query_len = 513
    seed = 2026
    update_percent = 7

    table_keys_cpu = np.empty((req_num, TABLE_SIZE), dtype=np.int32)
    table_states_cpu = np.empty((req_num, TABLE_SIZE), dtype=np.int32)
    query_cpu = np.empty((req_num, query_len), dtype=np.int32)
    new_states_cpu = np.empty((req_num, query_len), dtype=np.int32)

    for req_id in range(req_num):
        table_keys_cpu[req_id] = rng.permutation(np.arange(TABLE_SIZE, dtype=np.int32)).astype(np.int32)
        table_states_cpu[req_id] = (
            table_keys_cpu[req_id].astype(np.int64) * 13 + 1000 * req_id + 17
        ).astype(np.int32)
        query_cpu[req_id] = rng.choice(
            np.arange(TABLE_SIZE, dtype=np.int32), size=query_len, replace=True).astype(np.int32)
        query_cpu[req_id, req_id::101] = (
            3000 + 100 * req_id + np.arange(len(query_cpu[req_id, req_id::101]))
        ).astype(np.int32)
        new_states_cpu[req_id] = (
            200000 + 10000 * req_id + np.arange(query_len, dtype=np.int32)
        ).astype(np.int32)

    expected_out, expected_states = expected_lookup_update(
        table_keys_cpu, table_states_cpu, query_cpu, new_states_cpu, seed, update_percent)

    table_keys = torch.from_numpy(table_keys_cpu).npu()
    table_states = torch.from_numpy(table_states_cpu.copy()).npu()
    query_keys = torch.from_numpy(query_cpu).npu()
    new_states = torch.from_numpy(new_states_cpu).npu()

    out = hbm_lookup_update.lookup_random_update(
        table_keys,
        table_states,
        query_keys,
        new_states,
        seed=seed,
        update_percent=update_percent,
        block_dim=8,
        not_found=NOT_FOUND,
        do_update=True,
    )
    torch.npu.synchronize()

    out_cpu = out.cpu().numpy()
    states_after_cpu = table_states.cpu().numpy()

    assert tuple(out_cpu.shape) == (req_num, query_len)
    np.testing.assert_array_equal(out_cpu, expected_out)
    np.testing.assert_array_equal(states_after_cpu, expected_states)

    lookup_only_out = hbm_lookup_update.lookup_only(
        table_keys,
        torch.from_numpy(table_states_cpu.copy()).npu(),
        query_keys,
        block_dim=8,
        not_found=NOT_FOUND,
    )
    torch.npu.synchronize()
    np.testing.assert_array_equal(lookup_only_out.cpu().numpy(), expected_out)

    update_only_states = torch.from_numpy(table_states_cpu.copy()).npu()
    hbm_lookup_update.update_only(
        table_keys,
        update_only_states,
        query_keys,
        new_states,
        seed=seed,
        update_percent=update_percent,
        block_dim=8,
    )
    torch.npu.synchronize()
    np.testing.assert_array_equal(update_only_states.cpu().numpy(), expected_states)

    print("PASS: multi-req lookup output matches per-req pre-update states")
    print("PASS: multi-req update_only matches combined lookup/update state mutation")
    print(f"PASS: multi-req table_states updated at {len(update_positions(query_len, seed, update_percent))} positions per req")


def main():
    torch.npu.set_device(0)
    rng = np.random.default_rng(20260528)

    run_single_req_case(rng)
    run_multi_req_case(rng)


if __name__ == "__main__":
    main()
