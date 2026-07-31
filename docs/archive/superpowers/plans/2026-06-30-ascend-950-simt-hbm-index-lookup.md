# Ascend 950 SIMT HBM Index Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new standalone PTA/SIMT ASU HBM index lookup operator for Ascend 950 without modifying the existing AIV custom-op sources.

**Architecture:** Add `pta-ops/asu_hbm_index_lookup_simt` as an isolated package. The package builds an Ascend C SIMT kernel shared library and a PyTorch extension wrapper that validates tensors, allocates `slot_out`, launches the kernel on the current NPU stream, and returns `slot_out`.

**Tech Stack:** Ascend C/CANN 9.0 SIMT, torch/torch_npu C++ extension, CMake, pytest, NumPy reference validation.

---

## File Structure

- Create `pta-ops/asu_hbm_index_lookup_simt/README.md`: usage, build commands, runtime validation commands, and contract.
- Create `pta-ops/asu_hbm_index_lookup_simt/CMakeLists.txt`: standalone CMake project building the kernel shared library and PyTorch extension.
- Create `pta-ops/asu_hbm_index_lookup_simt/build.sh`: repeatable build entrypoint for Ascend 950.
- Create `pta-ops/asu_hbm_index_lookup_simt/include/asu_hbm_index_lookup_simt_constants.h`: shared constants and exported launcher signature.
- Create `pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_kernel.cpp`: Ascend C SIMT kernel and `asu_hbm_index_lookup_simt_do` launcher.
- Create `pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_torch.cpp`: PyTorch extension wrapper and tensor validation.
- Create `pta-ops/asu_hbm_index_lookup_simt/scripts/validate_lookup_simt.py`: runtime validation against `ops/scripts/asu_hbm_index_common.py`.
- Create `pta-ops/tests/test_lookup_simt_static_layout.py`: static tests for package layout, symbols, constants, and old-source isolation.

No existing `ascend-ops/asu_hbm_index_lookup/*` or `ops/asu_hbm_index_lookup_aiv.cpp` files are modified.

---

### Task 1: Static Tests For The New PTA/SIMT Package

**Files:**
- Create: `pta-ops/tests/test_lookup_simt_static_layout.py`

- [ ] **Step 1: Write the failing static layout tests**

Create `pta-ops/tests/test_lookup_simt_static_layout.py` with this content:

```python
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
    assert "asc_atomic_cas" in kernel
    assert "ASU_HBM_CLAIMING" in kernel
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
```

- [ ] **Step 2: Run the static tests and verify they fail**

Run:

```bash
pytest -q pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
FAILED pta-ops/tests/test_lookup_simt_static_layout.py::test_lookup_simt_pta_tree_exists
```

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add pta-ops/tests/test_lookup_simt_static_layout.py
git commit -m "test: add simt lookup pta static layout checks"
```

---

### Task 2: Package Skeleton And Build Scaffolding

**Files:**
- Create: `pta-ops/asu_hbm_index_lookup_simt/README.md`
- Create: `pta-ops/asu_hbm_index_lookup_simt/CMakeLists.txt`
- Create: `pta-ops/asu_hbm_index_lookup_simt/build.sh`
- Create: `pta-ops/asu_hbm_index_lookup_simt/include/asu_hbm_index_lookup_simt_constants.h`

- [ ] **Step 1: Add the constants and launcher header**

Create `pta-ops/asu_hbm_index_lookup_simt/include/asu_hbm_index_lookup_simt_constants.h`:

```cpp
#ifndef ASU_HBM_INDEX_LOOKUP_SIMT_CONSTANTS_H
#define ASU_HBM_INDEX_LOOKUP_SIMT_CONSTANTS_H

#include <cstdint>

constexpr uint32_t ASU_HBM_INDEX_SIZE = 128U * 1024U;
constexpr uint32_t ASU_HBM_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t ASU_HBM_FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t ASU_HBM_QUERY_COUNT = 2U * 1024U;
constexpr uint32_t ASU_HBM_SIMT_THREADS = 256U;
constexpr int32_t ASU_HBM_NOT_FOUND = -1;
constexpr int32_t ASU_HBM_CLAIMING = -2;

extern "C" void asu_hbm_index_lookup_simt_do(void* stream,
                                             void* index,
                                             void* slot_to_index,
                                             void* free_slots,
                                             void* free_head,
                                             void* query_index,
                                             void* slot_out,
                                             uint32_t req_num);

#endif
```

- [ ] **Step 2: Add the standalone CMake project**

Create `pta-ops/asu_hbm_index_lookup_simt/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(asu_hbm_index_lookup_simt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "Ascend950" CACHE STRING "Ascend SoC version")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "CANN ascend-toolkit path")
set(PYTHON_BIN "python3" CACHE STRING "Python executable")

if(NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  elseif(DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_INSTALL_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
  message(FATAL_ERROR "ascendc_kernel_cmake does not exist; check ASCEND_CANN_PACKAGE_PATH")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

execute_process(
  COMMAND ${PYTHON_BIN} -c "import torch, pybind11; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
)

ascendc_library(asu_hbm_index_lookup_simt_kernel SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/asu_hbm_index_lookup_simt_kernel.cpp
)
ascendc_compile_definitions(asu_hbm_index_lookup_simt_kernel PRIVATE
  -DASCENDC_DUMP=0
)

pybind11_add_module(asu_hbm_index_lookup_simt
  ${CMAKE_CURRENT_SOURCE_DIR}/src/asu_hbm_index_lookup_simt_torch.cpp
)
target_include_directories(asu_hbm_index_lookup_simt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(asu_hbm_index_lookup_simt PRIVATE
  ${TORCH_LIBRARIES}
  asu_hbm_index_lookup_simt_kernel
)
set_target_properties(asu_hbm_index_lookup_simt PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
```

- [ ] **Step 3: Add the build script**

Create `pta-ops/asu_hbm_index_lookup_simt/build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"

if [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
  -DPYTHON_BIN="${PYTHON_BIN}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j

echo "Built PTA SIMT lookup package in ${BUILD_DIR}"
```

- [ ] **Step 4: Add the package README**

Create `pta-ops/asu_hbm_index_lookup_simt/README.md`:

```markdown
# ASU HBM Index Lookup SIMT PTA

This package adds a standalone PTA implementation of ASU HBM index lookup for Ascend 950.

It does not modify the existing `ascend-ops/asu_hbm_index_lookup` CANN custom-op implementation or the prototype `ops/asu_hbm_index_lookup_aiv.cpp` source.

## Contract

Inputs:

- `index`: int32 NPU tensor with at least `req_num * 128K` elements.
- `slot_to_index`: int32 NPU tensor with at least `req_num * 10K` elements.
- `free_slots`: int32 NPU tensor with at least `req_num * 2K` elements.
- `free_head`: int32 NPU tensor with at least `req_num` elements.
- `query_index`: int32 NPU tensor with at least `req_num * 2K` elements.
- `req_num`: positive integer.

Output:

- `slot_out`: int32 NPU tensor with the same shape as `query_index`.

The operator updates `index`, `slot_to_index`, and `free_head` in place when a query misses.

## Build

```bash
cd ASU-Ascend/pta-ops/asu_hbm_index_lookup_simt
SOC_VERSION=Ascend950 bash build.sh
```

## Validate

```bash
PYTHONPATH=build:../../ops/scripts python3 scripts/validate_lookup_simt.py --req-num 2 --pattern mixed
```

The validation script compares the PTA SIMT output with the Python reference in `ops/scripts/asu_hbm_index_common.py`.
```

- [ ] **Step 5: Make the build script executable and run static tests**

Run:

```bash
chmod +x pta-ops/asu_hbm_index_lookup_simt/build.sh
pytest -q pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
FAILED pta-ops/tests/test_lookup_simt_static_layout.py::test_lookup_simt_constants_and_launcher_symbols
```

- [ ] **Step 6: Commit the skeleton**

Run:

```bash
git add pta-ops/asu_hbm_index_lookup_simt/README.md \
  pta-ops/asu_hbm_index_lookup_simt/CMakeLists.txt \
  pta-ops/asu_hbm_index_lookup_simt/build.sh \
  pta-ops/asu_hbm_index_lookup_simt/include/asu_hbm_index_lookup_simt_constants.h
git commit -m "feat: add simt lookup pta package skeleton"
```

---

### Task 3: SIMT Kernel And Launcher

**Files:**
- Create: `pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_kernel.cpp`

- [ ] **Step 1: Add the SIMT kernel source**

Create `pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_kernel.cpp`:

```cpp
#include "asu_hbm_index_lookup_simt_constants.h"

#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t QueryOffsetForThread(uint32_t thread_id)
{
    return thread_id;
}

__aicore__ inline void ClaimMisses(__gm__ int32_t* index,
                                   __gm__ int32_t* query_index,
                                   uint32_t index_req_base,
                                   uint32_t query_req_base)
{
    uint32_t thread_id = static_cast<uint32_t>(Simt::GetThreadIdx());
    uint32_t thread_num = static_cast<uint32_t>(Simt::GetThreadNum());

    for (uint32_t q = QueryOffsetForThread(thread_id); q < ASU_HBM_QUERY_COUNT; q += thread_num) {
        int32_t token = query_index[query_req_base + q];
        __gm__ int32_t* slot_addr = index + index_req_base + static_cast<uint32_t>(token);
        int32_t slot = *slot_addr;
        if (slot == ASU_HBM_NOT_FOUND) {
            (void)asc_atomic_cas(slot_addr, ASU_HBM_NOT_FOUND, ASU_HBM_CLAIMING);
        }
    }
}

__aicore__ inline void AllocateClaimedMisses(__gm__ int32_t* index,
                                             __gm__ int32_t* slot_to_index,
                                             __gm__ int32_t* free_slots,
                                             __gm__ int32_t* free_head,
                                             __gm__ int32_t* query_index,
                                             uint32_t index_req_base,
                                             uint32_t slot_req_base,
                                             uint32_t free_req_base,
                                             uint32_t query_req_base,
                                             uint32_t req_id)
{
    int32_t head = free_head[req_id];
    for (uint32_t q = 0; q < ASU_HBM_QUERY_COUNT; ++q) {
        int32_t token = query_index[query_req_base + q];
        __gm__ int32_t* slot_addr = index + index_req_base + static_cast<uint32_t>(token);
        if (*slot_addr == ASU_HBM_CLAIMING) {
            int32_t slot = free_slots[free_req_base + static_cast<uint32_t>(head)];
            ++head;
            slot_to_index[slot_req_base + static_cast<uint32_t>(slot)] = token;
            *slot_addr = slot;
        }
    }
    free_head[req_id] = head;
}

__aicore__ inline void WriteOutput(__gm__ int32_t* index,
                                   __gm__ int32_t* query_index,
                                   __gm__ int32_t* slot_out,
                                   uint32_t index_req_base,
                                   uint32_t query_req_base)
{
    uint32_t thread_id = static_cast<uint32_t>(Simt::GetThreadIdx());
    uint32_t thread_num = static_cast<uint32_t>(Simt::GetThreadNum());

    for (uint32_t q = thread_id; q < ASU_HBM_QUERY_COUNT; q += thread_num) {
        int32_t token = query_index[query_req_base + q];
        slot_out[query_req_base + q] = index[index_req_base + static_cast<uint32_t>(token)];
    }
}

__aicore__ inline void ProcessRequest(__gm__ int32_t* index,
                                      __gm__ int32_t* slot_to_index,
                                      __gm__ int32_t* free_slots,
                                      __gm__ int32_t* free_head,
                                      __gm__ int32_t* query_index,
                                      __gm__ int32_t* slot_out)
{
    uint32_t req_id = get_block_idx();
    uint32_t index_req_base = req_id * ASU_HBM_INDEX_SIZE;
    uint32_t slot_req_base = req_id * ASU_HBM_SLOT_COUNT;
    uint32_t free_req_base = req_id * ASU_HBM_FREE_SLOT_COUNT;
    uint32_t query_req_base = req_id * ASU_HBM_QUERY_COUNT;

    ClaimMisses(index, query_index, index_req_base, query_req_base);
    Simt::SyncBlock();

    if (Simt::GetThreadIdx() == 0) {
        AllocateClaimedMisses(index,
                              slot_to_index,
                              free_slots,
                              free_head,
                              query_index,
                              index_req_base,
                              slot_req_base,
                              free_req_base,
                              query_req_base,
                              req_id);
    }
    Simt::SyncBlock();

    WriteOutput(index, query_index, slot_out, index_req_base, query_req_base);
}

}  // namespace

extern "C" __global__ __aicore__ void asu_hbm_index_lookup_simt_kernel(GM_ADDR index,
                                                                        GM_ADDR slot_to_index,
                                                                        GM_ADDR free_slots,
                                                                        GM_ADDR free_head,
                                                                        GM_ADDR query_index,
                                                                        GM_ADDR slot_out)
{
    ProcessRequest(reinterpret_cast<__gm__ int32_t*>(index),
                   reinterpret_cast<__gm__ int32_t*>(slot_to_index),
                   reinterpret_cast<__gm__ int32_t*>(free_slots),
                   reinterpret_cast<__gm__ int32_t*>(free_head),
                   reinterpret_cast<__gm__ int32_t*>(query_index),
                   reinterpret_cast<__gm__ int32_t*>(slot_out));
}

extern "C" void asu_hbm_index_lookup_simt_do(void* stream,
                                             void* index,
                                             void* slot_to_index,
                                             void* free_slots,
                                             void* free_head,
                                             void* query_index,
                                             void* slot_out,
                                             uint32_t req_num)
{
#ifndef ASCENDC_CPU_DEBUG
    asu_hbm_index_lookup_simt_kernel<<<req_num, ASU_HBM_SIMT_THREADS, stream>>>(
        reinterpret_cast<GM_ADDR>(index),
        reinterpret_cast<GM_ADDR>(slot_to_index),
        reinterpret_cast<GM_ADDR>(free_slots),
        reinterpret_cast<GM_ADDR>(free_head),
        reinterpret_cast<GM_ADDR>(query_index),
        reinterpret_cast<GM_ADDR>(slot_out));
#endif
}
```

- [ ] **Step 2: Run static tests and verify kernel tokens pass**

Run:

```bash
pytest -q pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
FAILED pta-ops/tests/test_lookup_simt_static_layout.py::test_lookup_simt_pta_tree_exists
```

The remaining failure is for `src/asu_hbm_index_lookup_simt_torch.cpp` or `scripts/validate_lookup_simt.py`, because they are not created yet.

- [ ] **Step 3: Commit the kernel**

Run:

```bash
git add pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_kernel.cpp
git commit -m "feat: add ascend 950 simt lookup kernel"
```

---

### Task 4: PyTorch PTA Wrapper

**Files:**
- Create: `pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_torch.cpp`

- [ ] **Step 1: Add the PyTorch extension wrapper**

Create `pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_torch.cpp`:

```cpp
#include "asu_hbm_index_lookup_simt_constants.h"

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace {

void CheckInt32NpuContiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().is_privateuseone(), name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckNumelAtLeast(const at::Tensor& tensor, const char* name, int64_t expected)
{
    TORCH_CHECK(tensor.numel() >= expected,
                name,
                " must have at least ",
                expected,
                " elements, got ",
                tensor.numel());
}

}  // namespace

at::Tensor asu_hbm_index_lookup_simt(at::Tensor index,
                                     at::Tensor slot_to_index,
                                     at::Tensor free_slots,
                                     at::Tensor free_head,
                                     at::Tensor query_index,
                                     int64_t req_num)
{
    CheckInt32NpuContiguous(index, "index");
    CheckInt32NpuContiguous(slot_to_index, "slot_to_index");
    CheckInt32NpuContiguous(free_slots, "free_slots");
    CheckInt32NpuContiguous(free_head, "free_head");
    CheckInt32NpuContiguous(query_index, "query_index");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");

    CheckNumelAtLeast(index, "index", req_num * static_cast<int64_t>(ASU_HBM_INDEX_SIZE));
    CheckNumelAtLeast(slot_to_index, "slot_to_index", req_num * static_cast<int64_t>(ASU_HBM_SLOT_COUNT));
    CheckNumelAtLeast(free_slots, "free_slots", req_num * static_cast<int64_t>(ASU_HBM_FREE_SLOT_COUNT));
    CheckNumelAtLeast(free_head, "free_head", req_num);
    CheckNumelAtLeast(query_index, "query_index", req_num * static_cast<int64_t>(ASU_HBM_QUERY_COUNT));

    at::Tensor slot_out = at::empty_like(query_index);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    asu_hbm_index_lookup_simt_do(stream,
                                 index.data_ptr(),
                                 slot_to_index.data_ptr(),
                                 free_slots.data_ptr(),
                                 free_head.data_ptr(),
                                 query_index.data_ptr(),
                                 slot_out.data_ptr(),
                                 static_cast<uint32_t>(req_num));
    return slot_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("asu_hbm_index_lookup_simt",
          &asu_hbm_index_lookup_simt,
          "ASU HBM index lookup and miss allocation using Ascend 950 SIMT PTA");
}
```

- [ ] **Step 2: Run static tests and verify the runtime script is the only missing file**

Run:

```bash
pytest -q pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
FAILED pta-ops/tests/test_lookup_simt_static_layout.py::test_lookup_simt_pta_tree_exists
```

The missing file named in the assertion is `scripts/validate_lookup_simt.py`.

- [ ] **Step 3: Commit the wrapper**

Run:

```bash
git add pta-ops/asu_hbm_index_lookup_simt/src/asu_hbm_index_lookup_simt_torch.cpp
git commit -m "feat: add simt lookup pytorch wrapper"
```

---

### Task 5: Runtime Validation Script

**Files:**
- Create: `pta-ops/asu_hbm_index_lookup_simt/scripts/validate_lookup_simt.py`

- [ ] **Step 1: Add the validation script**

Create `pta-ops/asu_hbm_index_lookup_simt/scripts/validate_lookup_simt.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
REPO_ROOT = PKG_DIR.parents[1]
OPS_SCRIPT_DIR = REPO_ROOT / "ops" / "scripts"
if str(OPS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_SCRIPT_DIR))

from asu_hbm_index_common import (  # noqa: E402
    QUERY_COUNT,
    expected_lookup_allocate,
    make_index_case,
    require_numpy,
    require_runtime,
    to_npu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ASU HBM SIMT PTA lookup.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--req-num", type=int, default=2)
    parser.add_argument("--pattern", choices=("hit", "miss", "mixed"), default="mixed")
    parser.add_argument("--module-dir", type=Path, default=PKG_DIR / "build")
    return parser.parse_args()


def import_module(module_dir: Path):
    module_path = module_dir.expanduser().resolve()
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
    import asu_hbm_index_lookup_simt

    return asu_hbm_index_lookup_simt


def make_npu_tensors(torch, case):
    return {
        "index": to_npu(torch, case.index),
        "slot_to_index": to_npu(torch, case.slot_to_index),
        "free_slots": to_npu(torch, case.free_slots),
        "free_head": to_npu(torch, case.free_head),
        "query_index": to_npu(torch, case.query_index),
    }


def main() -> None:
    args = parse_args()
    if args.req_num <= 0:
        raise ValueError("--req-num must be greater than 0")

    np = require_numpy()
    torch = require_runtime(args.device)
    simt_module = import_module(args.module_dir)
    case = make_index_case(args.req_num, args.pattern)
    expected = expected_lookup_allocate(case)
    tensors = make_npu_tensors(torch, case)

    slot_out = simt_module.asu_hbm_index_lookup_simt(
        tensors["index"],
        tensors["slot_to_index"],
        tensors["free_slots"],
        tensors["free_head"],
        tensors["query_index"],
        args.req_num,
    )
    torch.npu.synchronize()

    np.testing.assert_array_equal(slot_out.cpu().numpy(), expected.slot_out)
    np.testing.assert_array_equal(tensors["index"].cpu().numpy(), expected.index)
    np.testing.assert_array_equal(tensors["slot_to_index"].cpu().numpy(), expected.slot_to_index)
    np.testing.assert_array_equal(tensors["free_head"].cpu().numpy(), expected.free_head)

    unique_misses = int(expected.free_head.sum())
    print(
        "PASS lookup simt: req_num={} pattern={} query_count={} unique_misses={}".format(
            args.req_num, args.pattern, QUERY_COUNT, unique_misses
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the script executable and run static tests**

Run:

```bash
chmod +x pta-ops/asu_hbm_index_lookup_simt/scripts/validate_lookup_simt.py
pytest -q pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
4 passed
```

- [ ] **Step 3: Commit the validation script**

Run:

```bash
git add pta-ops/asu_hbm_index_lookup_simt/scripts/validate_lookup_simt.py
git commit -m "test: add simt lookup runtime validation script"
```

---

### Task 6: Local Verification And Final Documentation Check

**Files:**
- Modify: `pta-ops/asu_hbm_index_lookup_simt/README.md`

- [ ] **Step 1: Run all local tests available without CANN**

Run:

```bash
pytest -q ascend-ops/tests/test_static_layout.py pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
8 passed
```

- [ ] **Step 2: Check the new files for trailing whitespace**

Run:

```bash
git diff --check
```

Expected result: no output.

- [ ] **Step 3: Add runtime build notes to the README**

Append this section to `pta-ops/asu_hbm_index_lookup_simt/README.md`:

````markdown
## Runtime Environment

The SIMT kernel targets Ascend 950 with CANN 9.0 PTA support. This development workspace may not contain CANN headers or Ascend 950 hardware, so local CI covers static layout checks. On a runtime host, build with `SOC_VERSION=Ascend950 bash build.sh`, then run `scripts/validate_lookup_simt.py` for hit, miss, and mixed patterns.

```bash
PYTHONPATH=build:../../ops/scripts python3 scripts/validate_lookup_simt.py --req-num 1 --pattern hit
PYTHONPATH=build:../../ops/scripts python3 scripts/validate_lookup_simt.py --req-num 2 --pattern mixed
PYTHONPATH=build:../../ops/scripts python3 scripts/validate_lookup_simt.py --req-num 8 --pattern miss
```
````

- [ ] **Step 4: Re-run static tests after README change**

Run:

```bash
pytest -q pta-ops/tests/test_lookup_simt_static_layout.py
```

Expected result:

```text
4 passed
```

- [ ] **Step 5: Commit the README update**

Run:

```bash
git add pta-ops/asu_hbm_index_lookup_simt/README.md
git commit -m "docs: document simt lookup runtime validation"
```

- [ ] **Step 6: Report final verification**

Run:

```bash
git status --short
git log --oneline -5
```

Expected result:

```text
git status --short
```

prints no tracked or untracked changes. `git log --oneline -5` shows the implementation commits from this plan.

---

## Self-Review

Spec coverage:

- New standalone PTA/SIMT package: Tasks 2 through 5.
- Existing source isolation: Task 1 static test and file structure statement.
- Lookup-and-allocate semantics: Task 3 kernel and Task 5 reference validation.
- Duplicate miss correctness: Task 3 `ASU_HBM_CLAIMING` plus serial request-local allocation.
- Host validation: Task 4 wrapper.
- Static and runtime testing: Tasks 1, 5, and 6.

Placeholder scan:

- The plan contains no placeholder tokens and every code-producing step includes exact file content.

Type consistency:

- Constants use the `ASU_HBM_*` prefix across tests, kernel, wrapper, and validation.
- The exported launcher is consistently named `asu_hbm_index_lookup_simt_do`.
- The Python extension function is consistently named `asu_hbm_index_lookup_simt`.
