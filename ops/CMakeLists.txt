cmake_minimum_required(VERSION 3.16)

set(ASU_HBM_INDEX_OP "" CACHE STRING "Operator to build: lookup_aiv")
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC version, e.g. Ascend910B1/B2/B3/B4")
set(NPU_ARCH "dav-2201" CACHE STRING "Ascend NPU arch for ASC host compilation, e.g. dav-2201 for Ascend 910B")
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

if(ASU_HBM_INDEX_OP STREQUAL "maintain_aicpu")
  list(APPEND CMAKE_PREFIX_PATH "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
  find_package(ASC REQUIRED)
  find_package(AICPU REQUIRED)
  project(asu_hbm_index_ops LANGUAGES ASC AICPU CXX)

  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/asu_hbm_index_maintain_aicpu.cpp
    PROPERTIES LANGUAGE ASC
  )

  add_library(asu_hbm_index_maintain_aicpu SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/asu_hbm_index_maintain_aicpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/asu_hbm_index_maintain_aicpu_kernel.aicpu
  )
  set_target_properties(asu_hbm_index_maintain_aicpu PROPERTIES
    LINKER_LANGUAGE ASC
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
  target_compile_options(asu_hbm_index_maintain_aicpu PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
  )
  target_include_directories(asu_hbm_index_maintain_aicpu PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/aicpu_api
  )
  target_link_libraries(asu_hbm_index_maintain_aicpu PRIVATE
    ascendc_runtime
    profapi
    ascendalog
    ascendcl
    runtime
    c_sec
    mmpa
    error_manager
    ascend_dump
  )
elseif(ASU_HBM_INDEX_OP STREQUAL "lookup_aiv")
  project(asu_hbm_index_ops LANGUAGES CXX)

  if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist; check ASCEND_CANN_PACKAGE_PATH")
  endif()

  include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

  ascendc_library(asu_hbm_index_lookup_aiv SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/asu_hbm_index_lookup_aiv.cpp
  )
  ascendc_compile_definitions(asu_hbm_index_lookup_aiv PRIVATE
    -DASCENDC_DUMP=0
  )
else()
  message(FATAL_ERROR "Unsupported ASU_HBM_INDEX_OP=${ASU_HBM_INDEX_OP}; use lookup_aiv or maintain_aicpu")
endif()
