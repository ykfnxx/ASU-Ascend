#!/usr/bin/env python3
"""Collect and synchronously parse a single-op Ascend 950 SIMT profile."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lookup_simt_common import (
    PKG_DIR,
    assert_runtime_result,
    call_lookup,
    estimate_state_bytes,
    expected_result,
    load_kernel,
    require_runtime,
    to_npu_state,
)

from python.random_workload import (  # type: ignore[import-not-found]
    QUERY_COUNT,
    SLOT_COUNT,
    make_random_case,
    make_random_query_row,
    validate_hit_count,
)


DEFAULT_HIT_COUNT = 1843


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile only asu_hbm_index_lookup_simt on Ascend 950. "
            "Tensor construction, H2D copies, and warmup are outside the "
            "profile; every captured call has an exact hit count and fresh "
            "randomly distributed misses."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=PKG_DIR / "build")
    parser.add_argument(
        "--library-path",
        type=Path,
        default=None,
        help="optional path to libasu_hbm_index_lookup_simt_kernel.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=1)
    parser.add_argument(
        "--hit-count",
        type=int,
        default=DEFAULT_HIT_COUNT,
        help=f"exact hits per {QUERY_COUNT}-token request (default: {DEFAULT_HIT_COUNT})",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument(
        "--export-type",
        choices=("db", "text"),
        default="db",
        help="torch-npu parsed output format (default: db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty directory for raw, parsed, and manifest outputs",
    )
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.req_num <= 0:
        raise ValueError("--req-num must be positive")
    if args.device < 0:
        raise ValueError("--device cannot be negative")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.profile_iterations <= 0:
        raise ValueError("--profile-iterations must be positive")
    validate_hit_count(args.hit_count)


def prepare_output_dir(output_dir: Path) -> tuple[Path, Path, Path]:
    run_root = output_dir.expanduser().resolve()
    if run_root.exists() and not run_root.is_dir():
        raise RuntimeError(f"--output-dir is not a directory: {run_root}")
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(
            f"--output-dir must be empty to avoid mixing profiles: {run_root}"
        )
    raw_root = run_root / "raw"
    parsed_root = run_root / "parsed"
    raw_root.mkdir(parents=True, exist_ok=True)
    parsed_root.mkdir(parents=True, exist_ok=True)
    return run_root, raw_root, parsed_root


def verify_one_state(
    np: Any,
    torch: Any,
    kernel: Any,
    req_num: int,
    hit_count: int,
    seed: int,
) -> None:
    case = make_random_case(
        np,
        req_num,
        hit_count,
        seed=seed,
        case_id=0,
    )
    expected = expected_result(np, case)
    state = to_npu_state(torch, case)
    outputs = call_lookup(kernel, torch, state, req_num)
    torch.npu.synchronize()
    assert_runtime_result(np, state, outputs, expected)


def preload_states(
    np: Any,
    torch: Any,
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any]]:
    states = [
        to_npu_state(
            torch,
            make_random_case(
                np,
                args.req_num,
                args.hit_count,
                seed=args.seed,
                case_id=case_id,
            ),
        )
        for case_id in range(1, 1 + args.warmup + args.profile_iterations)
    ]
    torch.npu.synchronize()
    return states[: args.warmup], states[args.warmup :]


def run_warmup(
    torch: Any,
    kernel: Any,
    states: list[Any],
    req_num: int,
) -> None:
    retained_outputs = [
        call_lookup(kernel, torch, state, req_num) for state in states
    ]
    torch.npu.synchronize()
    if len(retained_outputs) != len(states):
        raise RuntimeError("warmup output retention failed")


def create_experimental_config(torch_npu: Any, export_type: str) -> Any:
    profiler_export_type = (
        torch_npu.profiler.ExportType.Db
        if export_type == "db"
        else torch_npu.profiler.ExportType.Text
    )
    return torch_npu.profiler._ExperimentalConfig(
        export_type=profiler_export_type,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )


def profile_states(
    torch: Any,
    torch_npu: Any,
    kernel: Any,
    states: list[Any],
    args: argparse.Namespace,
    raw_root: Path,
) -> None:
    trace_name = (
        f"asu_lookup_simt_req{args.req_num}_hit{args.hit_count}"
    )
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(raw_root),
        worker_name=trace_name,
        analyse_flag=True,
        async_mode=False,
    )
    retained_outputs = []
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=create_experimental_config(
            torch_npu, args.export_type
        ),
        on_trace_ready=handler,
    ):
        for state in states:
            retained_outputs.append(
                call_lookup(kernel, torch, state, args.req_num)
            )
        torch.npu.synchronize()

    if len(retained_outputs) != args.profile_iterations:
        raise RuntimeError("profile did not retain every operator output")


def is_raw_profile_dir(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    return any(
        child.is_dir()
        and (child.name == "FRAMEWORK" or child.name.startswith("PROF_"))
        for child in children
    )


def discover_raw_profile(raw_root: Path) -> Path:
    candidates = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_dir() and is_raw_profile_dir(path)
    )
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            "expected exactly one raw profile under "
            f"{raw_root}, found {len(candidates)}: {rendered}"
        )
    return candidates[0]


def copy_parsed_results(
    raw_profile: Path,
    parsed_root: Path,
    export_type: str,
) -> list[Path]:
    source_root = raw_profile / "ASCEND_PROFILER_OUTPUT"
    if not source_root.is_dir():
        raise RuntimeError(
            "torch-npu did not create parsed ASCEND_PROFILER_OUTPUT: "
            f"{source_root}"
        )

    source_files = sorted(
        path for path in source_root.rglob("*") if path.is_file()
    )
    expected_suffixes = {".db"} if export_type == "db" else {".csv", ".json"}
    if not any(
        path.suffix.lower() in expected_suffixes for path in source_files
    ):
        raise RuntimeError(
            "parsed profiler output contains none of the expected files: "
            + ", ".join(sorted(expected_suffixes))
        )

    copied = []
    destination_root = parsed_root / "ASCEND_PROFILER_OUTPUT"
    for source in source_files:
        destination = destination_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    for pattern in ("profiler_info*.json", "profiler_metadata.json"):
        for source in raw_profile.glob(pattern):
            destination = parsed_root / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    return sorted(copied)


def git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def device_name(torch: Any, device_id: int) -> str | None:
    try:
        return str(torch.npu.get_device_name(device_id))
    except (AttributeError, RuntimeError):
        return None


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    torch: Any,
    torch_npu: Any,
    library_path: Path,
    raw_profile: Path,
    parsed_files: list[Path],
) -> None:
    first_profile_case_id = 1 + args.warmup
    query = make_random_query_row(
        args.hit_count,
        seed=args.seed,
        case_id=first_profile_case_id,
        req_id=0,
    )
    miss_positions = [
        position
        for position, token in enumerate(query)
        if token >= SLOT_COUNT
    ]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operator": "asu_hbm_index_lookup_simt",
        "configuration": {
            "req_num": args.req_num,
            "query_count_per_request": QUERY_COUNT,
            "hit_count_per_request": args.hit_count,
            "miss_count_per_request": QUERY_COUNT - args.hit_count,
            "hit_ratio": args.hit_count / QUERY_COUNT,
            "seed": args.seed,
            "random_unique_miss_tokens": True,
            "randomized_miss_positions": True,
            "first_profile_case_id": first_profile_case_id,
            "request0_miss_position_sample": miss_positions[:32],
            "warmup_iterations": args.warmup,
            "profile_iterations": args.profile_iterations,
            "profiler_level": "Level1",
            "aic_metrics": "PipeUtilization",
            "export_type": args.export_type,
            "direct_parse": True,
        },
        "environment": {
            "device_id": args.device,
            "device_name": device_name(torch, args.device),
            "torch_version": getattr(torch, "__version__", None),
            "torch_npu_version": getattr(torch_npu, "__version__", None),
            "kernel_library": str(library_path),
            "git_commit": git_commit(PKG_DIR),
        },
        "outputs": {
            "raw_profile": str(raw_profile),
            "parsed_files": [str(output) for output in parsed_files],
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_root, raw_root, parsed_root = prepare_output_dir(args.output_dir)
    np, torch, torch_npu = require_runtime(args.device)
    kernel = load_kernel(args.library_path, args.build_dir)
    miss_count = QUERY_COUNT - args.hit_count
    state_mib = estimate_state_bytes(args.req_num) / 1024.0 / 1024.0

    print(
        "config: req_num={} hit_count={} miss_count={} hit_ratio={:.6f} "
        "warmup={} profile_iterations={} seed={}".format(
            args.req_num,
            args.hit_count,
            miss_count,
            args.hit_count / QUERY_COUNT,
            args.warmup,
            args.profile_iterations,
            args.seed,
        )
    )
    print(f"library: {kernel.path}")
    print(f"output: {run_root}")
    print(
        "preload estimate: {:.2f} MiB/state, {:.2f} MiB total "
        "input/state/workspace tensors".format(
            state_mib,
            state_mib * (args.warmup + args.profile_iterations),
        )
    )

    if not args.no_verify:
        verify_one_state(
            np,
            torch,
            kernel,
            args.req_num,
            args.hit_count,
            args.seed,
        )
        print("verify: PASS")

    warmup, captured = preload_states(np, torch, args)
    run_warmup(torch, kernel, warmup, args.req_num)
    print("warmup: PASS")
    profile_states(
        torch,
        torch_npu,
        kernel,
        captured,
        args,
        raw_root,
    )
    raw_profile = discover_raw_profile(raw_root)
    parsed_files = copy_parsed_results(
        raw_profile,
        parsed_root,
        args.export_type,
    )
    manifest_path = run_root / "manifest.json"
    write_manifest(
        manifest_path,
        args=args,
        torch=torch,
        torch_npu=torch_npu,
        library_path=kernel.path,
        raw_profile=raw_profile,
        parsed_files=parsed_files,
    )
    print(f"profile: PASS raw={raw_profile}")
    print(f"parsed: {parsed_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
