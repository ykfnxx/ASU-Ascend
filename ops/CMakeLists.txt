cmake_minimum_required(VERSION 3.16)
project(asu_ascend_ops)

# This file mirrors the vLLM-Ascend custom-op source layout. It is intended to
# be copied into a CANN custom-op build tree that already defines the standard
# targets used below: op_host_aclnn, optiling, and opsproto.

add_subdirectory(asu_resolve_kv_slots/op_host)
