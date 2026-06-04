add_ops_compile_options(
        OP_NAME AsuResolveKvSlots
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

set(asu_resolve_kv_slots_depends experimental/asu_resolve_kv_slots PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        asu_resolve_kv_slots_def.cpp
)

target_sources(optiling PRIVATE
        asu_resolve_kv_slots_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            asu_resolve_kv_slots_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        asu_resolve_kv_slots_proto.cpp
)
