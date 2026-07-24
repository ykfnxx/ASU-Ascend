import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = REPO_ROOT / "pta-ops" / "asu_hbm_index_lookup_simt"


def read(path: str) -> str:
    return (PKG_DIR / path).read_text(encoding="utf-8")


def test_lookup_simt_pta_tree_exists():
    required_files = [
        "README.md",
        "CMakeLists.txt",
        "build.sh",
        "include/asu_hbm_index_lookup_simt_constants.h",
        "src/asu_hbm_index_lookup_simt_kernel.cpp",
        "src/asu_hbm_index_lookup_simt_torch.cpp",
        "python/lookup_lru_reference.py",
        "python/random_workload.py",
        "scripts/check_soc_version.sh",
        "scripts/build_lookup_simt.sh",
        "scripts/bench_lookup_simt.py",
        "scripts/lookup_simt_common.py",
        "scripts/profile_lookup_simt.py",
        "scripts/validate_lookup_simt.py",
        "tests/test_reference.py",
        "tests/stubs/kernel_operator.h",
        "tests/stubs/simt_api/common_functions.h",
        "tests/stubs/simt_api/device_atomic_functions.h",
        "tests/stubs/simt_api/device_sync_functions.h",
    ]

    for file_name in required_files:
        assert (PKG_DIR / file_name).is_file(), file_name


def test_lookup_simt_keeps_original_sources_isolated():
    status_sensitive_files = [
        "ascend-ops/asu_hbm_index_lookup/op_kernel/asu_hbm_index_lookup.cpp",
        "ascend-ops/asu_hbm_index_lookup/asu_hbm_index_lookup_torch_adpt.h",
        "ops/asu_hbm_index_lookup_aiv.cpp",
    ]

    for relative_path in status_sensitive_files:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_lookup_simt_constants_and_launcher_symbols():
    constants = read("include/asu_hbm_index_lookup_simt_constants.h")
    kernel = read("src/asu_hbm_index_lookup_simt_kernel.cpp")
    wrapper = read("src/asu_hbm_index_lookup_simt_torch.cpp")

    for token in [
        "ASU_HBM_INDEX_SIZE = 128U * 1024U",
        "ASU_HBM_SLOT_COUNT = 10U * 1024U",
        "ASU_HBM_QUERY_COUNT = 2U * 1024U",
        "ASU_HBM_NOT_FOUND = -1",
        "ASU_HBM_CLAIMING = -2",
        "ASU_HBM_SIMT_THREADS = 256U",
        "ASU_HBM_WORKSPACE_STRIDE",
    ]:
        assert token in constants

    assert 'extern "C" __global__ __aicore__ void asu_hbm_index_lookup_simt_kernel' in kernel
    assert "asu_hbm_index_lookup_simt_do" in kernel
    assert '#include "simt_api/common_functions.h"' in kernel
    assert '#include "simt_api/device_sync_functions.h"' in kernel
    assert '#include "simt_api/device_atomic_functions.h"' in kernel
    assert "__simt_vf__" in kernel
    assert "asc_vf_call" in kernel
    assert "asc_syncthreads" in kernel
    assert "asc_atomic_cas" in kernel
    assert "ASU_HBM_CLAIMING" in kernel
    assert "hit_slots" in kernel
    assert "evictable_slots" in kernel
    assert "stale_count" in kernel
    assert "victim_token" in kernel
    assert "miss_mask" in kernel
    assert "workspace_size" in wrapper
    assert "std::tuple<at::Tensor, at::Tensor>" in wrapper
    assert "at::kBool" in wrapper
    assert "threadIdx.x" in kernel
    assert "PYBIND11_MODULE" in wrapper
    assert "asu_hbm_index_lookup_simt" in wrapper
    assert '#include "torch_npu/csrc/core/npu/NPUGuard.h"' in wrapper
    assert "c10_npu::NPUGuard npu_guard(device)" in wrapper
    assert "OptionalNPUGuard" not in wrapper


def test_lookup_simt_closes_allocation_eviction_and_lru_state():
    header = read("include/asu_hbm_index_lookup_simt_constants.h")
    kernel = read("src/asu_hbm_index_lookup_simt_kernel.cpp")
    wrapper = read("src/asu_hbm_index_lookup_simt_torch.cpp")
    readme = read("README.md")
    validate = read("scripts/validate_lookup_simt.py")
    bench = read("scripts/bench_lookup_simt.py")
    workload = read("python/random_workload.py")

    assert "void* free_head" not in header
    assert "at::Tensor free_head" not in wrapper
    assert "free_slots" not in header
    assert "free_slots" not in wrapper
    assert "free_slots" not in kernel
    assert "token_to_slot" in header
    assert "slot_to_token" in header
    assert "lru_slots" in header
    assert "miss_mask" in header
    assert "req_token_to_slot[static_cast<uint32_t>(victim_token)]" in kernel
    assert "req_slot_to_token[victim_slot] = token" in kernel
    assert "untouched stale slots + newly allocated miss slots" in kernel
    assert "host_cache" not in kernel
    assert "device_buffer" not in kernel
    assert "expected_result" in validate
    assert "assert_runtime_result" in validate
    assert "expected_result" in bench
    assert "make_random_case" in workload
    assert "free_head" not in readme


def test_lookup_simt_kernel_is_valid_cxx_with_api_stubs():
    compiler = shutil.which("c++")
    if compiler is None:
        return
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-DASCENDC_CPU_DEBUG",
            f"-I{PKG_DIR / 'include'}",
            f"-I{PKG_DIR / 'tests' / 'stubs'}",
            "-fsyntax-only",
            str(PKG_DIR / "src" / "asu_hbm_index_lookup_simt_kernel.cpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_lookup_simt_build_files_target_ascend_950_pta():
    cmake = read("CMakeLists.txt")
    build = read("build.sh")
    build_driver = read("scripts/build_lookup_simt.sh")
    readme = read("README.md")

    assert "ascendc_library" in cmake
    assert "pybind11_add_module" in cmake
    assert (
        "ascendc_include_directories("
        "asu_hbm_index_lookup_simt_kernel PRIVATE"
    ) in cmake
    assert (
        "target_include_directories("
        "asu_hbm_index_lookup_simt_kernel PRIVATE"
    ) not in cmake
    assert "ASU_HBM_INDEX_LOOKUP_SIMT_INCLUDE_DIR" in cmake
    assert "\ninclude_directories(" not in cmake
    assert 'set(Python3_EXECUTABLE "${PYTHON_BIN}")' in cmake
    assert (
        'set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)'
        in cmake
    )
    assert 'SOC_VERSION ""' in cmake
    assert 'SOC_VERSION="${SOC_VERSION:-}"' in build
    assert "--soc-version" in build_driver
    assert '-DSOC_VERSION="${SOC_VERSION_ARG}"' in build_driver
    assert "Unsupported Ascend 950 SOC_VERSION" not in cmake
    assert "unsupported Ascend 950 SOC_VERSION" not in build
    assert "unsupported Ascend 950 SOC_VERSION" not in build_driver
    assert "import torch_npu" in build_driver
    assert "built module:" in build_driver
    assert "set(CMAKE_SKIP_RPATH FALSE)" in cmake
    assert cmake.index("include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)") < cmake.index(
        "set(CMAKE_SKIP_RPATH FALSE)"
    )
    assert 'BUILD_RPATH "$ORIGIN;$ORIGIN/lib;' in cmake
    assert 'INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;' in cmake
    assert "PTA" in readme
    assert "Ascend 950" in readme


def test_lookup_simt_soc_checker_preserves_runtime_value():
    checker = PKG_DIR / "scripts" / "check_soc_version.sh"
    completed = subprocess.run(
        [
            "bash",
            str(checker),
            "--runtime-soc",
            "runtime-returned-soc",
            "--value-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "runtime-returned-soc"
    assert "torch.npu.get_device_name" in read("scripts/check_soc_version.sh")


def test_lookup_simt_random_workload_has_exact_hits_and_shuffled_misses():
    workload = read("python/random_workload.py")

    for token in [
        "validate_hit_count",
        "rng.sample(range(SLOT_COUNT), hit_count)",
        "rng.sample(range(SLOT_COUNT, INDEX_SIZE), miss_count)",
        "rng.shuffle(query)",
        "case_id",
        "req_id",
    ]:
        assert token in workload


def test_lookup_simt_benchmark_preloads_fresh_npu_states():
    bench = read("scripts/bench_lookup_simt.py")

    for token in [
        "default=100",
        "default=50",
        "--hit-count",
        "DEFAULT_HIT_COUNT = 1843",
        "make_random_case",
        "preload_states",
        "to_npu_state",
        "torch.npu.synchronize()",
        "torch.npu.Event",
        "time.perf_counter()",
        "elapsed_time",
        "verify_one_state",
        "outputs.append",
        "event_ns_per_query",
        "randomized_miss_positions",
    ]:
        assert token in bench


def test_lookup_simt_profile_is_single_op_and_directly_parsed():
    profile = read("scripts/profile_lookup_simt.py")

    for token in [
        "--hit-count",
        "make_random_case",
        "preload_states",
        "torch_npu.profiler.profile",
        "torch_npu.profiler.tensorboard_trace_handler",
        "analyse_flag=True",
        "async_mode=False",
        "ProfilerLevel.Level1",
        "outputs",
        "ASCEND_PROFILER_OUTPUT",
        "manifest.json",
        "randomized_miss_positions",
    ]:
        assert token in profile
