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
        "scripts/bench_lookup_simt.py",
        "scripts/validate_lookup_simt.py",
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
        "ASU_HBM_FREE_SLOT_COUNT = 2U * 1024U",
        "ASU_HBM_QUERY_COUNT = 2U * 1024U",
        "ASU_HBM_NOT_FOUND = -1",
        "ASU_HBM_CLAIMING = -2",
        "ASU_HBM_SIMT_THREADS = 256U",
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
    assert "threadIdx.x" in kernel
    assert "blockIdx.x" in kernel
    assert "PYBIND11_MODULE" in wrapper
    assert "asu_hbm_index_lookup_simt" in wrapper


def test_lookup_simt_build_files_target_ascend_950_pta():
    cmake = read("CMakeLists.txt")
    build = read("build.sh")
    readme = read("README.md")

    assert "ascendc_library" in cmake
    assert "pybind11_add_module" in cmake
    assert "Ascend950" in build
    assert "SOC_VERSION" in build
    assert "PTA" in readme
    assert "Ascend 950" in readme


def test_lookup_simt_benchmark_preloads_fresh_npu_states():
    bench = read("scripts/bench_lookup_simt.py")

    for token in [
        "default=100",
        "default=50",
        "default=0.10",
        "preload_benchmark_states",
        "to_npu(torch, case.index)",
        "torch.npu.synchronize()",
        "time.perf_counter()",
        "elapsed_time",
        "expected_lookup_allocate",
        "unique_misses_per_req",
        "outputs.append",
    ]:
        assert token in bench
