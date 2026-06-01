import argparse
import csv
import ctypes
import os
import sys
import time
from contextlib import nullcontext


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL_LIB_NAME = "libhbm_lookup_update_kernels_npu.so"
INDEX_SIZE = 128 * 1024
INT32_BYTES = 4
PATTERNS = {"fixed", "sequential", "stride", "block", "hotset", "random"}


def parse_csv_ints(text):
    return [int(x) for x in str(text).split(",") if x]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile lookup access patterns to diagnose scalar GM memory pressure.")
    parser.add_argument("--req-num", type=int, default=50,
                        help="Number of request tables.")
    parser.add_argument("--query-len", type=int, default=2048,
                        help="Number of query keys per request.")
    parser.add_argument("--block-dim", type=int, default=64,
                        help="Kernel blockDim for lookup.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations per pattern.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations per pattern.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base random seed.")
    parser.add_argument("--device", type=int, default=0,
                        help="NPU device id.")
    parser.add_argument("--kv-block-size", type=int, default=128,
                        help="KV cache block size used for reporting unique block count.")
    parser.add_argument("--hotset-size", default="128,1024,8192",
                        help="Hotset sizes for hotset patterns, comma-separated.")
    parser.add_argument("--patterns", default="fixed,sequential,block,hotset,random",
                        help="Patterns to run: fixed,sequential,stride,block,hotset,random.")
    parser.add_argument("--build-dir", default=os.environ.get("BUILD_DIR", os.path.join(ROOT, "build")),
                        help="Directory containing the built hbm_lookup_update extension.")
    parser.add_argument("--profile-dir", default=None,
                        help="If set, collect torch_npu profiler for --profile-pattern.")
    parser.add_argument("--profile-pattern", default="random",
                        help="Pattern to profile when --profile-dir is set.")
    parser.add_argument("--profile-hotset-size", type=int, default=8192,
                        help="Hotset size used if --profile-pattern=hotset.")
    parser.add_argument("--profiler-level", default="level2",
                        choices=("none", "level0", "level1", "level2"),
                        help="torch_npu profiler detail level.")
    parser.add_argument("--aic-metrics", default="memory",
                        choices=("none", "pipe", "arithmetic", "memory", "ub", "l2cache", "resource"),
                        help="AI Core metrics set when supported by torch_npu profiler.")
    parser.add_argument("--profile-rows", type=int, default=20,
                        help="Rows to print from profiler CSV/text tables.")
    return parser.parse_args()


def import_runtime(build_dir):
    build_dir = os.path.abspath(build_dir)
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

    import numpy as np
    import torch
    import torch_npu

    kernel_lib = os.path.join(build_dir, "lib", KERNEL_LIB_NAME)
    if os.path.exists(kernel_lib):
        ctypes.CDLL(kernel_lib, mode=ctypes.RTLD_GLOBAL)

    import hbm_lookup_update
    return np, torch, torch_npu, hbm_lookup_update


def make_table_tensors(np, torch, req_num):
    token_ids = np.arange(INDEX_SIZE, dtype=np.int32)
    table_keys_cpu = np.empty((req_num, INDEX_SIZE), dtype=np.int32)
    table_states_cpu = np.empty((req_num, INDEX_SIZE), dtype=np.int32)

    for req_id in range(req_num):
        table_keys_cpu[req_id] = token_ids
        table_states_cpu[req_id] = (
            token_ids.astype(np.int64) + req_id * INDEX_SIZE
        ).astype(np.int32)

    return (
        torch.from_numpy(table_keys_cpu).npu(),
        torch.from_numpy(table_states_cpu).npu(),
    )


def make_query_cpu(np, pattern, req_num, query_len, seed, hotset_size, kv_block_size):
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern}")

    query = np.empty((req_num, query_len), dtype=np.int32)
    base = np.arange(query_len, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for req_id in range(req_num):
        if pattern == "fixed":
            query[req_id].fill((seed + req_id * 257) % INDEX_SIZE)
        elif pattern == "sequential":
            query[req_id] = ((base + seed + req_id * 257) % INDEX_SIZE).astype(np.int32)
        elif pattern == "stride":
            query[req_id] = ((base * 17 + seed + req_id * 257) % INDEX_SIZE).astype(np.int32)
        elif pattern == "block":
            block_count = max(1, INDEX_SIZE // kv_block_size)
            query[req_id] = ((base * 17 + seed + req_id * 257) % block_count).astype(np.int32)
        elif pattern == "hotset":
            size = min(max(1, hotset_size), INDEX_SIZE)
            query[req_id] = rng.integers(0, size, size=query_len, dtype=np.int32)
        elif pattern == "random":
            query[req_id] = rng.integers(0, INDEX_SIZE, size=query_len, dtype=np.int32)
    return query


def query_stats(np, query_cpu, kv_block_size):
    per_req_unique = [np.unique(row).size for row in query_cpu]
    per_req_blocks = [np.unique(row // kv_block_size).size for row in query_cpu]
    return {
        "unique_keys_avg": sum(per_req_unique) / len(per_req_unique),
        "unique_keys_max": max(per_req_unique),
        "unique_blocks_avg": sum(per_req_blocks) / len(per_req_blocks),
        "unique_blocks_max": max(per_req_blocks),
    }


def make_event_pair(torch):
    event_cls = getattr(torch.npu, "Event", None)
    if event_cls is None:
        return None, None
    try:
        return event_cls(enable_timing=True), event_cls(enable_timing=True)
    except TypeError:
        return event_cls(), event_cls()


def time_loop(torch, iters, body):
    torch.npu.synchronize()
    start_event, end_event = make_event_pair(torch)
    if start_event is not None:
        start_event.record()
    host_start = time.perf_counter()

    for i in range(iters):
        body(i)

    if end_event is not None:
        end_event.record()
    torch.npu.synchronize()
    host_elapsed = time.perf_counter() - host_start

    device_ms = None
    if start_event is not None and end_event is not None:
        try:
            device_ms = start_event.elapsed_time(end_event)
        except Exception:
            device_ms = None
    return host_elapsed, device_ms


def enum_value(enum_cls, name_map, user_value):
    if enum_cls is None:
        return None
    attr_name = name_map.get(user_value)
    if attr_name is None:
        return None
    return getattr(enum_cls, attr_name, None)


def profiler_context(torch_npu, profile_dir, profiler_level, aic_metrics):
    if not profile_dir:
        return nullcontext()

    os.makedirs(profile_dir, exist_ok=True)
    profiler = torch_npu.profiler
    kwargs = {
        "activities": [
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.NPU,
        ],
        "with_stack": False,
        "profile_memory": False,
        "with_modules": False,
        "on_trace_ready": profiler.tensorboard_trace_handler(profile_dir),
    }

    experimental_config_cls = getattr(profiler, "_ExperimentalConfig", None)
    if experimental_config_cls is not None:
        level = enum_value(
            getattr(profiler, "ProfilerLevel", None),
            {
                "none": "Level_none",
                "level0": "Level0",
                "level1": "Level1",
                "level2": "Level2",
            },
            profiler_level,
        )
        metrics = enum_value(
            getattr(profiler, "AiCMetrics", None),
            {
                "none": "AiCoreNone",
                "pipe": "PipeUtilization",
                "arithmetic": "ArithmeticUtilization",
                "memory": "Memory",
                "ub": "MemoryUB",
                "l2cache": "L2Cache",
                "resource": "ResourceConflictRatio",
            },
            aic_metrics,
        )
        export_type = getattr(getattr(profiler, "ExportType", None), "Text", None)

        config_kwargs = {
            "msprof_tx": False,
            "l2_cache": aic_metrics == "l2cache",
            "op_attr": False,
            "data_simplification": True,
            "record_op_args": False,
        }
        if export_type is not None:
            config_kwargs["export_type"] = export_type
        if level is not None:
            config_kwargs["profiler_level"] = level
        if metrics is not None:
            config_kwargs["aic_metrics"] = metrics
        kwargs["experimental_config"] = experimental_config_cls(**config_kwargs)

    return profiler.profile(**kwargs)


def run_lookup(hbm_lookup_update, table_keys, table_states, query_keys, block_dim):
    return hbm_lookup_update.lookup_only(
        table_keys, table_states, query_keys, block_dim=block_dim, not_found=-1)


def bench_pattern(np, torch, hbm_lookup_update, table_keys, table_states, args,
                  pattern, hotset_size):
    query_cpu = make_query_cpu(
        np, pattern, args.req_num, args.query_len, args.seed,
        hotset_size, args.kv_block_size)
    stats = query_stats(np, query_cpu, args.kv_block_size)
    query_keys = torch.from_numpy(query_cpu).npu()

    for _ in range(args.warmup):
        run_lookup(hbm_lookup_update, table_keys, table_states, query_keys, args.block_dim)
    torch.npu.synchronize()

    host_elapsed, device_ms = time_loop(
        torch,
        args.iters,
        lambda _i: run_lookup(hbm_lookup_update, table_keys, table_states, query_keys, args.block_dim),
    )

    return query_cpu, stats, host_elapsed, device_ms


def format_pattern(pattern, hotset_size):
    if pattern == "hotset":
        return f"hotset{hotset_size}"
    return pattern


def print_result(pattern_name, stats, host_elapsed, device_ms, args):
    queries = args.req_num * args.query_len
    host_ms = host_elapsed * 1000.0 / args.iters
    if device_ms is None:
        device_ms_per_iter = None
        qps = queries * args.iters / host_elapsed
        global_ns = host_ms * 1_000_000.0 / queries
        core_ns = global_ns * args.block_dim
        state_gbps = 0.0
        approx_gbps = 0.0
    else:
        device_ms_per_iter = device_ms / args.iters
        device_s = device_ms_per_iter / 1000.0
        qps = queries / device_s
        global_ns = device_ms_per_iter * 1_000_000.0 / queries
        core_ns = global_ns * args.block_dim
        state_gbps = queries * INT32_BYTES / device_s / 1e9
        approx_gbps = queries * INT32_BYTES * 3 / device_s / 1e9

    device_text = "nan" if device_ms_per_iter is None else f"{device_ms_per_iter:.6f}"
    print(
        f"{pattern_name}\t{args.req_num}\t{args.query_len}\t{args.block_dim}\t"
        f"{stats['unique_keys_avg']:.1f}\t{stats['unique_keys_max']}\t"
        f"{stats['unique_blocks_avg']:.1f}\t{stats['unique_blocks_max']}\t"
        f"{host_ms:.6f}\t{device_text}\t{qps:.3f}\t"
        f"{global_ns:.3f}\t{core_ns:.3f}\t{state_gbps:.3f}\t{approx_gbps:.3f}")


def profile_one_pattern(np, torch, torch_npu, hbm_lookup_update, table_keys,
                        table_states, args):
    pattern = args.profile_pattern
    hotset_size = args.profile_hotset_size
    query_cpu = make_query_cpu(
        np, pattern, args.req_num, args.query_len, args.seed,
        hotset_size, args.kv_block_size)
    query_keys = torch.from_numpy(query_cpu).npu()

    for _ in range(args.warmup):
        run_lookup(hbm_lookup_update, table_keys, table_states, query_keys, args.block_dim)
    torch.npu.synchronize()

    with profiler_context(torch_npu, args.profile_dir, args.profiler_level, args.aic_metrics) as prof:
        time_loop(
            torch,
            args.iters,
            lambda _i: run_lookup(hbm_lookup_update, table_keys, table_states, query_keys, args.block_dim),
        )
        if prof is not None and hasattr(prof, "step"):
            prof.step()

    print(f"profile_dir={args.profile_dir}")
    print_profile_rows(args.profile_dir, args.profile_rows)


def read_profile_rows(path):
    rows = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not name.endswith((".csv", ".tsv", ".txt")):
                continue
            file_path = os.path.join(root, name)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                first_line = text.splitlines()[0] if text else ""
                if "Name" not in first_line or "hbm_lookup" not in text:
                    continue
                delimiter = "\t" if "\t" in first_line else ","
                reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
                for row in reader:
                    row_text = " ".join(str(v) for v in row.values())
                    if "hbm_lookup" in row_text:
                        row["_file"] = file_path
                        rows.append(row)
    return rows


def print_profile_rows(profile_dir, limit):
    rows = read_profile_rows(profile_dir)
    if not rows:
        print("profile_rows=unavailable")
        print("profile_hint=check profiler output manually under profile_dir")
        return

    fields = [
        "Name", "Type", "Duration(us)", "Block Dim",
        "aiv_time(us)", "aiv_total_cycles",
        "aiv_vec_time(us)", "aiv_vec_ratio",
        "aiv_scalar_time(us)", "aiv_scalar_ratio",
        "aiv_mte2_time(us)", "aiv_mte2_ratio",
        "aiv_mte3_time(us)", "aiv_mte3_ratio",
        "aiv_icache_miss_rate",
    ]
    print("profile_rows")
    print("\t".join(["file"] + fields))
    for row in rows[:limit]:
        values = [os.path.relpath(row.get("_file", ""), profile_dir)]
        values.extend(row.get(field, "") for field in fields)
        print("\t".join(values))


def main():
    args = parse_args()
    np, torch, torch_npu, hbm_lookup_update = import_runtime(args.build_dir)
    torch.npu.set_device(args.device)

    table_keys, table_states = make_table_tensors(np, torch, args.req_num)
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    for pattern in patterns + ([args.profile_pattern] if args.profile_dir else []):
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern: {pattern}")
    hotset_sizes = parse_csv_ints(args.hotset_size)

    print("pattern\treq_num\tquery_len\tblock_dim\tunique_keys_avg\tunique_keys_max\t"
          "unique_kv_blocks_avg\tunique_kv_blocks_max\thost_ms_per_iter\t"
          "device_ms_per_iter\tlookup_qps_device\tglobal_ns_per_query\t"
          "core_ns_per_query\tstate_load_gbps\tapprox_query_state_out_gbps")
    for pattern in patterns:
        sizes = hotset_sizes if pattern == "hotset" else [0]
        for hotset_size in sizes:
            _query_cpu, stats, host_elapsed, device_ms = bench_pattern(
                np, torch, hbm_lookup_update, table_keys, table_states,
                args, pattern, hotset_size)
            print_result(format_pattern(pattern, hotset_size), stats, host_elapsed, device_ms, args)

    if args.profile_dir:
        profile_one_pattern(
            np, torch, torch_npu, hbm_lookup_update, table_keys, table_states, args)


if __name__ == "__main__":
    main()
