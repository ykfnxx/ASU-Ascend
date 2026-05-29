import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.environ.get("BUILD_DIR", os.path.join(ROOT, "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import hbm_lookup_update  # noqa: E402


def main():
    torch.npu.set_device(0)
    table_size = 2048
    req_num = int(os.environ.get("REQ_NUM", "4"))
    query_len = int(os.environ.get("QUERY_LEN", "2048"))
    block_dim = int(os.environ.get("BLOCK_DIM", "8"))
    iters = int(os.environ.get("ITERS", "1000"))
    warmup = int(os.environ.get("WARMUP", "50"))
    update_percent = int(os.environ.get("UPDATE_PERCENT", "5"))

    rng = np.random.default_rng(1234)
    table_keys_cpu = np.empty((req_num, table_size), dtype=np.int32)
    table_states_cpu = np.empty((req_num, table_size), dtype=np.int32)
    query_cpu = np.empty((req_num, query_len), dtype=np.int32)
    new_states_cpu = np.empty((req_num, query_len), dtype=np.int32)
    for req_id in range(req_num):
        table_keys_cpu[req_id] = rng.permutation(np.arange(table_size, dtype=np.int32)).astype(np.int32)
        table_states_cpu[req_id] = (
            table_keys_cpu[req_id].astype(np.int64) * 3 + 1000 * req_id + 1
        ).astype(np.int32)
        query_cpu[req_id] = rng.choice(
            np.arange(table_size, dtype=np.int32), size=query_len, replace=True).astype(np.int32)
        new_states_cpu[req_id] = (
            777000 + 10000 * req_id + np.arange(query_len, dtype=np.int32)
        ).astype(np.int32)

    table_keys = torch.from_numpy(table_keys_cpu).npu()
    table_states = torch.from_numpy(table_states_cpu.copy()).npu()
    query_keys = torch.from_numpy(query_cpu).npu()
    new_states = torch.from_numpy(new_states_cpu).npu()

    for i in range(warmup):
        hbm_lookup_update.lookup_random_update(
            table_keys, table_states, query_keys, new_states,
            seed=i, update_percent=update_percent, block_dim=block_dim,
            not_found=-1, do_update=True,
        )
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for i in range(iters):
        hbm_lookup_update.lookup_random_update(
            table_keys, table_states, query_keys, new_states,
            seed=10000 + i, update_percent=update_percent, block_dim=block_dim,
            not_found=-1, do_update=True,
        )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    lookups = iters * req_num * query_len
    updates = iters * req_num * ((query_len * update_percent) // 100)
    print(f"req_num={req_num} query_len={query_len} block_dim={block_dim} update_percent={update_percent}")
    print(f"iters={iters} elapsed={elapsed:.6f}s")
    print(f"lookup_qps={lookups / elapsed:.3f}")
    print(f"update_qps={updates / elapsed:.3f}")


if __name__ == "__main__":
    main()
