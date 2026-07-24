cmake_minimum_required(VERSION 3.16)

project(asu_hbm_index_lookup_simt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "Ascend950" CACHE STRING "Ascend SoC version")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "CANN ascend-toolkit path")
set(PYTHON_BIN "python3" CACHE STRING "Python executable")

if(NOT SOC_VERSION STREQUAL "Ascend950")
  message(FATAL_ERROR "asu_hbm_index_lookup_simt supports Ascend950 only")
endif()

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

set(Python3_EXECUTABLE "${PYTHON_BIN}")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${PYTHON_BIN} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query torch.utils.cmake_prefix_path")
endif()
string(REGEX MATCH "[^\r\n]+$" TORCH_CMAKE_PREFIX "${TORCH_CMAKE_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

execute_process(
  COMMAND ${PYTHON_BIN} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query pybind11 cmake directory")
endif()
string(REGEX MATCH "[^\r\n]+$" PYBIND11_CMAKE_DIR "${PYBIND11_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND ${PYTHON_BIN} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  RESULT_VARIABLE TORCH_CXX11_ABI_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CXX11_ABI_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query torch CXX11 ABI")
endif()
string(REGEX MATCH "[^\r\n]+$" TORCH_CXX11_ABI "${TORCH_CXX11_ABI}")

execute_process(
  COMMAND ${PYTHON_BIN} -c "import importlib.metadata as m; print(m.distribution('torch-npu').locate_file('torch_npu'))"
  OUTPUT_VARIABLE TORCH_NPU_ROOT
  RESULT_VARIABLE TORCH_NPU_ROOT_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_NPU_ROOT_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query torch-npu package path")
endif()
string(REGEX MATCH "[^\r\n]+$" TORCH_NPU_ROOT "${TORCH_NPU_ROOT}")

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
  ${TORCH_NPU_ROOT}/include
  ${TORCH_NPU_ROOT}/..
  ${ASCEND_CANN_PACKAGE_PATH}/runtime/include
)
target_compile_definitions(asu_hbm_index_lookup_simt PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)
target_link_directories(asu_hbm_index_lookup_simt PRIVATE
  ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
  ${TORCH_NPU_ROOT}/lib
)
target_link_libraries(asu_hbm_index_lookup_simt PRIVATE
  ${TORCH_LIBRARIES}
  asu_hbm_index_lookup_simt_kernel
  torch_npu
  ascendcl
)
set_target_properties(asu_hbm_index_lookup_simt PROPERTIES
  BUILD_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64;${TORCH_NPU_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64;${TORCH_NPU_ROOT}/lib"
)
