import argparse
import ctypes
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL_LIB_NAME = "libhbm_lookup_update_kernels_npu.so"
INDEX_SIZE = 128 * 1024


def parse_csv_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_args():
    parser = argparse.ArgumentParser(description="Quick benchmark for hbm lookup/update kernels.")
    parser.add_argument("--mode", choices=("lookup", "update", "both"), default=os.environ.get("MODE", "lookup"))
    parser.add_argument("--req-num", default=os.environ.get("REQ_NUM", "4"),
                        help="Request counts, comma-separated for sweeps.")
    parser.add_argument("--query-len", type=int, default=int(os.environ.get("QUERY_LEN", "2048")))
    parser.add_argument("--block-dim", default=os.environ.get("BLOCK_DIM", "8"),
                        help="Block dims, comma-separated for sweeps.")
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "100")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "10")))
    parser.add_argument("--update-percent", type=int, default=int(os.environ.get("UPDATE_PERCENT", "5")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1234")))
    parser.add_argument("--device", type=int, default=int(os.environ.get("DEVICE", "0")))
    parser.add_argument("--build-dir", default=os.environ.get("BUILD_DIR", os.path.join(ROOT, "build")))
    return parser.parse_args()


def import_runtime(build_dir):
    build_dir = os.path.abspath(build_dir)
    sys.path.insert(0, build_dir)

    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    kernel_lib = os.path.join(build_dir, "lib", KERNEL_LIB_NAME)
    if os.path.exists(kernel_lib):
        ctypes.CDLL(kernel_lib, mode=ctypes.RTLD_GLOBAL)

    import hbm_lookup_update
    return np, torch, hbm_lookup_update


def make_inputs(np, torch, req_num, query_len, seed):
    padded_query_len = ((query_len + 63) // 64) * 64
    if padded_query_len > INDEX_SIZE:
        raise ValueError("query_len must be <= 128K for the contiguous-key simulation")

    token_ids = np.arange(INDEX_SIZE, dtype=np.int32)
    table_keys_cpu = np.empty((req_num, INDEX_SIZE), dtype=np.int32)
    table_states_cpu = np.empty((req_num, INDEX_SIZE), dtype=np.int32)
    query_cpu = np.empty((req_num, query_len), dtype=np.int32)
    new_states_cpu = np.empty((req_num, query_len), dtype=np.int32)
    start_limit = INDEX_SIZE - padded_query_len + 1

    for req_id in range(req_num):
        table_keys_cpu[req_id] = token_ids
        table_states_cpu[req_id] = (
            token_ids.astype(np.int64) + req_id * INDEX_SIZE
        ).astype(np.int32)
        start = (seed + req_id * 257) % start_limit
        query_cpu[req_id] = start + np.arange(query_len, dtype=np.int32)
        new_states_cpu[req_id] = (
            777000 + 10000 * req_id + np.arange(query_len, dtype=np.int32)
        ).astype(np.int32)

    return (
        torch.from_numpy(table_keys_cpu).npu(),
        torch.from_numpy(table_states_cpu.copy()).npu(),
        torch.from_numpy(query_cpu).npu(),
        torch.from_numpy(new_states_cpu).npu(),
    )


def call_op(hbm_lookup_update, mode, tensors, block_dim, update_percent, seed):
    table_keys, table_states, query_keys, new_states = tensors
    if mode == "lookup":
        return hbm_lookup_update.lookup_only(
            table_keys, table_states, query_keys, block_dim=block_dim, not_found=-1)
    if mode == "update":
        hbm_lookup_update.update_only(
            table_keys, table_states, query_keys, new_states,
            seed=seed, update_percent=update_percent, block_dim=block_dim)
        return None
    return hbm_lookup_update.lookup_random_update(
        table_keys, table_states, query_keys, new_states,
        seed=seed, update_percent=update_percent, block_dim=block_dim,
        not_found=-1, do_update=True)


def bench_one(np, torch, hbm_lookup_update, args, mode, req_num, block_dim):
    tensors = make_inputs(np, torch, req_num, args.query_len, args.seed)
    for i in range(args.warmup):
        call_op(hbm_lookup_update, mode, tensors, block_dim, args.update_percent, args.seed + i)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for i in range(args.iters):
        call_op(hbm_lookup_update, mode, tensors, block_dim, args.update_percent, args.seed + 10000 + i)
    end.record()
    torch.npu.synchronize()
    device_ms = start.elapsed_time(end)

    lookup_count = args.iters * req_num * args.query_len if mode != "update" else 0
    update_count = args.iters * req_num * ((args.query_len * args.update_percent) // 100) if mode != "lookup" else 0
    elapsed = device_ms / 1000.0
    lookup_qps = lookup_count / elapsed if lookup_count else 0.0
    update_qps = update_count / elapsed if update_count else 0.0
    print(
        f"{mode}\t{req_num}\t{args.query_len}\t{block_dim}\t{args.update_percent}\t"
        f"{args.iters}\t{device_ms / args.iters:.6f}\t{lookup_qps:.3f}\t{update_qps:.3f}")


def main():
    args = parse_args()
    np, torch, hbm_lookup_update = import_runtime(args.build_dir)
    torch.npu.set_device(args.device)

    print("mode\treq_num\tquery_len\tblock_dim\tupdate_percent\titers\tdevice_ms_per_iter\tlookup_qps\tupdate_qps")
    for req_num in parse_csv_ints(args.req_num):
        for block_dim in parse_csv_ints(args.block_dim):
            bench_one(np, torch, hbm_lookup_update, args, args.mode, req_num, block_dim)


if __name__ == "__main__":
    main()
