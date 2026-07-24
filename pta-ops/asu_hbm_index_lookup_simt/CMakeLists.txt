cmake_minimum_required(VERSION 3.16)

project(asu_hbm_index_lookup_simt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 9.1's legacy ascendc_library merge command always forwards
# CMAKE_BUILD_TYPE to merge_mix_obj.sh.  An empty value leaves --build-type
# without an argument and the script exits before linking any device object.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(SOC_VERSION "" CACHE STRING "Exact Ascend SoC version")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "CANN ascend-toolkit path")

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

set(ASU_HBM_INDEX_LOOKUP_SIMT_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

ascendc_library(asu_hbm_index_lookup_simt_kernel SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/asu_hbm_index_lookup_simt_kernel.cpp
)
ascendc_include_directories(asu_hbm_index_lookup_simt_kernel PRIVATE
  ${ASU_HBM_INDEX_LOOKUP_SIMT_INCLUDE_DIR}
  ${ASCEND_CANN_PACKAGE_PATH}/include
)
ascendc_compile_definitions(asu_hbm_index_lookup_simt_kernel PRIVATE
  -DASCENDC_DUMP=0
)
