import argparse
import os
import sys
import time
from contextlib import nullcontext


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile hbm_lookup_update lookup-only and update-only kernels.")
    parser.add_argument("--mode", choices=("lookup", "update", "both"),
                        default="both", help="Which kernel path to run.")
    parser.add_argument("--req-num", type=int, default=4,
                        help="Number of request tables, R.")
    parser.add_argument("--query-len", type=int, default=2048,
                        help="Number of query keys per request, Q.")
    parser.add_argument("--block-dim", type=int, default=8,
                        help="Kernel blockDim for lookup and max update request parallelism.")
    parser.add_argument("--iters", type=int, default=200,
                        help="Timed/profiled iterations.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations before timing/profile.")
    parser.add_argument("--update-percent", type=int, default=5,
                        help="Percent of query positions selected by update_only.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base random seed for synthetic input and update positions.")
    parser.add_argument("--device", type=int, default=0,
                        help="NPU device id.")
    parser.add_argument("--build-dir", default=os.environ.get("BUILD_DIR", os.path.join(ROOT, "build")),
                        help="Directory containing the built hbm_lookup_update extension.")
    parser.add_argument("--profile-dir", default=None,
                        help="If set, write torch_npu profiler traces under this directory.")
    parser.add_argument("--profiler-level", default="level1",
                        choices=("none", "level0", "level1", "level2"),
                        help="torch_npu profiler detail level when --profile-dir is set.")
    parser.add_argument("--aic-metrics", default="none",
                        choices=("none", "pipe", "arithmetic", "memory", "ub", "l2cache", "resource"),
                        help="AI Core metrics set when supported by torch_npu profiler.")
    return parser.parse_args()


def import_runtime(build_dir):
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

    import numpy as np
    import torch
    import torch_npu
    import hbm_lookup_update

    return np, torch, torch_npu, hbm_lookup_update


def make_inputs(np, torch, req_num, query_len, seed):
    table_size = 2048
    rng = np.random.default_rng(seed)
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
    return table_keys, table_states, query_keys, new_states


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

    activities = [
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.NPU,
    ]
    kwargs = {
        "activities": activities,
        "with_stack": False,
        "profile_memory": False,
        "with_modules": False,
        "on_trace_ready": profiler.tensorboard_trace_handler(profile_dir),
    }

    experimental_config_cls = getattr(profiler, "_ExperimentalConfig", None)
    if experimental_config_cls is not None:
        export_type = enum_value(getattr(profiler, "ExportType", None), {"level": "Text"}, "level")
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

        config_kwargs = {
            "msprof_tx": False,
            "l2_cache": False,
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


def synchronize(torch):
    torch.npu.synchronize()


def warmup_lookup(hbm_lookup_update, torch, table_keys, table_states, query_keys, args):
    for _ in range(args.warmup):
        hbm_lookup_update.lookup_only(
            table_keys, table_states, query_keys,
            block_dim=args.block_dim, not_found=-1)
    synchronize(torch)


def measure_lookup(hbm_lookup_update, torch, table_keys, table_states, query_keys, args):
    start = time.perf_counter()
    for _ in range(args.iters):
        hbm_lookup_update.lookup_only(
            table_keys, table_states, query_keys,
            block_dim=args.block_dim, not_found=-1)
    synchronize(torch)
    return time.perf_counter() - start


def warmup_update(hbm_lookup_update, torch, table_keys, table_states, query_keys, new_states, args):
    for i in range(args.warmup):
        hbm_lookup_update.update_only(
            table_keys, table_states, query_keys, new_states,
            seed=args.seed + i, update_percent=args.update_percent,
            block_dim=args.block_dim)
    synchronize(torch)


def measure_update(hbm_lookup_update, torch, table_keys, table_states, query_keys, new_states, args):
    start = time.perf_counter()
    for i in range(args.iters):
        hbm_lookup_update.update_only(
            table_keys, table_states, query_keys, new_states,
            seed=args.seed + args.warmup + i,
            update_percent=args.update_percent, block_dim=args.block_dim)
    synchronize(torch)
    return time.perf_counter() - start


def print_result(mode, elapsed, args, profile_dir):
    lookup_count = args.iters * args.req_num * args.query_len
    update_count = args.iters * args.req_num * ((args.query_len * args.update_percent) // 100)
    work_count = lookup_count if mode == "lookup" else update_count
    qps_name = "lookup_qps" if mode == "lookup" else "update_qps"

    print(f"mode={mode}")
    print(
        f"req_num={args.req_num} query_len={args.query_len} "
        f"block_dim={args.block_dim} update_percent={args.update_percent}")
    print(f"iters={args.iters} elapsed={elapsed:.6f}s ms_per_iter={elapsed * 1000.0 / args.iters:.6f}")
    if work_count > 0:
        print(f"{qps_name}={work_count / elapsed:.3f}")
    else:
        print(f"{qps_name}=0.000")
    if profile_dir:
        print(f"profile_dir={profile_dir}")


def main():
    args = parse_args()
    np, torch, torch_npu, hbm_lookup_update = import_runtime(args.build_dir)
    torch.npu.set_device(args.device)

    modes = ("lookup", "update") if args.mode == "both" else (args.mode,)
    for mode in modes:
        table_keys, table_states, query_keys, new_states = make_inputs(
            np, torch, args.req_num, args.query_len, args.seed)
        mode_profile_dir = None
        if args.profile_dir:
            mode_profile_dir = os.path.join(args.profile_dir, mode) if args.mode == "both" else args.profile_dir

        if mode == "lookup":
            warmup_lookup(
                hbm_lookup_update, torch, table_keys, table_states, query_keys, args)
        else:
            warmup_update(
                hbm_lookup_update, torch, table_keys, table_states, query_keys, new_states, args)

        with profiler_context(torch_npu, mode_profile_dir, args.profiler_level, args.aic_metrics):
            if mode == "lookup":
                elapsed = measure_lookup(
                    hbm_lookup_update, torch, table_keys, table_states, query_keys, args)
            else:
                elapsed = measure_update(
                    hbm_lookup_update, torch, table_keys, table_states, query_keys, new_states, args)
        print_result(mode, elapsed, args, mode_profile_dir)


if __name__ == "__main__":
    main()
