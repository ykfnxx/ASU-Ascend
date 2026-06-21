cmake_minimum_required(VERSION 3.16)

project(asu_hbm_index_ascend_ops LANGUAGES CXX)

# Standalone source package for vLLM-Ascend style custom operators.
# When this directory is included from the vLLM-Ascend custom-op CMake
# context, these targets are already defined by csrc/CMakeLists.txt.
set(ASU_HBM_INDEX_ASCEND_OP_DIRS
    asu_hbm_index_lookup
    asu_hbm_index_maintain
)

if(TARGET op_host_aclnn AND TARGET optiling AND TARGET opsproto)
    foreach(ASU_HBM_INDEX_OP_DIR ${ASU_HBM_INDEX_ASCEND_OP_DIRS})
        add_subdirectory(${ASU_HBM_INDEX_OP_DIR}/op_host)
    endforeach()
else()
    message(STATUS "ASU HBM index ops are in vLLM-Ascend source format. "
                   "Include this directory from a vLLM-Ascend csrc custom-op build to compile.")
endif()
