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
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_VARIABLE TORCH_NPU_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(hbm_lookup_update MODULE pybind/pybind_hbm_lookup_update.cpp)
add_dependencies(hbm_lookup_update hbm_lookup_update_kernels_${RUN_MODE})

target_compile_features(hbm_lookup_update PRIVATE cxx_std_17)
target_compile_definitions(hbm_lookup_update PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)

target_include_directories(hbm_lookup_update PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_ROOT}/include
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
  BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64;${TORCH_NPU_ROOT}/lib"
  INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR};${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64;${TORCH_NPU_ROOT}/lib"
)

install(TARGETS hbm_lookup_update LIBRARY DESTINATION .)
