cmake_minimum_required(VERSION 3.16)
project(hbm_lookup_update LANGUAGES CXX)

set(RUN_MODE "npu" CACHE STRING "Only npu is supported by the pybind build")
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC version, e.g. Ascend910B1/B2/B3/B4")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "CANN ascend-toolkit path")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release/Debug" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "install prefix" FORCE)
endif()

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/hbm_lookup_update_kernel.cpp)
include(cmake/npu_lib.cmake)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query torch.utils.cmake_prefix_path")
endif()
string(REGEX MATCH "[^\r\n]+$" TORCH_CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query pybind11 cmake directory")
endif()
string(REGEX MATCH "[^\r\n]+$" PYBIND11_CMAKE_DIR "${PYBIND11_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  RESULT_VARIABLE TORCH_CXX11_ABI_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CXX11_ABI_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query torch CXX11 ABI")
endif()
string(REGEX MATCH "[^\r\n]+$" TORCH_CXX11_ABI "${TORCH_CXX11_ABI}")
if(NOT TORCH_CXX11_ABI MATCHES "^[01]$")
  message(FATAL_ERROR "Could not parse torch CXX11 ABI from Python output")
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import importlib.metadata as m; print(m.distribution('torch-npu').locate_file('torch_npu'))"
  OUTPUT_VARIABLE TORCH_NPU_ROOT
  RESULT_VARIABLE TORCH_NPU_ROOT_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_NPU_ROOT_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to query torch-npu package path")
endif()
string(REGEX MATCH "[^\r\n]+$" TORCH_NPU_ROOT "${TORCH_NPU_ROOT}")

pybind11_add_module(hbm_lookup_update MODULE pybind/pybind_hbm_lookup_update.cpp)
add_dependencies(hbm_lookup_update hbm_lookup_update_kernels_${RUN_MODE})

target_compile_features(hbm_lookup_update PRIVATE cxx_std_17)
target_compile_definitions(hbm_lookup_update PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)

target_include_directories(hbm_lookup_update PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_ROOT}/include
  ${TORCH_NPU_ROOT}/..
  ${ASCEND_CANN_PACKAGE_PATH}/runtime/include
)

target_link_directories(hbm_lookup_update PRIVATE
  ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
  ${TORCH_NPU_ROOT}/lib
)

target_link_libraries(hbm_lookup_update PRIVATE
  hbm_lookup_update_kernels_${RUN_MODE}
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
)

set_target_properties(hbm_lookup_update PROPERTIES
  BUILD_RPATH "$ORIGIN/lib;${CMAKE_CURRENT_BINARY_DIR}/lib;${CMAKE_CURRENT_BINARY_DIR};${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64;${TORCH_NPU_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/lib;$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64;${TORCH_NPU_ROOT}/lib"
)

install(TARGETS hbm_lookup_update LIBRARY DESTINATION .)
