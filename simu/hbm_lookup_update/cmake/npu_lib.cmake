if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
  message(FATAL_ERROR "ascendc_kernel_cmake does not exist; check ASCEND_CANN_PACKAGE_PATH")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(hbm_lookup_update_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})
ascendc_compile_definitions(hbm_lookup_update_kernels_${RUN_MODE} PRIVATE
  -DASCENDC_DUMP=0
)
