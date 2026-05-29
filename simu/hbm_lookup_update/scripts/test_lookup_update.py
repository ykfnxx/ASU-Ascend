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


def update_positions(query_len: int, seed: int, update_percent: int) -> List[int]:
    update_num = (query_len * update_percent) // 100
    if update_num <= 0:
        return []
    a = pick_coprime_a(u32(seed) ^ 0x9E3779B9, query_len)
    b = hash32(u32(seed) ^ 0x85EBCA6B) % query_len
    return [((a * t + b) % query_len) for t in range(update_num)]


def main():
    torch.npu.set_device(0)
    rng = np.random.default_rng(20260528)

    # table_keys is deliberately shuffled to prove this is comparison-based lookup,
    # not table_states[query_key] dense indexing.
    table_keys_cpu = rng.permutation(np.arange(TABLE_SIZE, dtype=np.int32)).astype(np.int32)
    table_states_cpu = (table_keys_cpu.astype(np.int64) * 10 + 7).astype(np.int32)

    query_len = 512
    query_cpu = rng.choice(np.arange(TABLE_SIZE, dtype=np.int32), size=query_len, replace=True).astype(np.int32)
    # Add several misses.
    query_cpu[::97] = (3000 + np.arange(len(query_cpu[::97]))).astype(np.int32)
    new_states_cpu = (100000 + np.arange(query_len, dtype=np.int32)).astype(np.int32)

    # Expected lookup output is the pre-update state.
    key_to_idx: Dict[int, int] = {int(k): int(i) for i, k in enumerate(table_keys_cpu)}
    expected_out = np.empty(query_len, dtype=np.int32)
    for i, qk in enumerate(query_cpu):
        idx = key_to_idx.get(int(qk))
        expected_out[i] = NOT_FOUND if idx is None else table_states_cpu[idx]

    expected_states = table_states_cpu.copy()
    seed = 42
    update_percent = 5
    for pos in update_positions(query_len, seed, update_percent):
        qk = int(query_cpu[pos])
        idx = key_to_idx.get(qk)
        if idx is not None:
            expected_states[idx] = new_states_cpu[pos]

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

    print("PASS: lookup output matches pre-update states")
    print(f"PASS: table_states updated at {len(update_positions(query_len, seed, update_percent))} query positions")
    print("sample query_keys:", query_cpu[:16].tolist())
    print("sample states_out:", out_cpu[:16].tolist())


if __name__ == "__main__":
    main()
