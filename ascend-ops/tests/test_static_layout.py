from pathlib import Path


ASCEND_OPS_DIR = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ASCEND_OPS_DIR / path).read_text(encoding="utf-8")


def test_vllm_ascend_style_tree_exists():
    required_files = [
        "CMakeLists.txt",
        "README.md",
        "torch_binding_asu_hbm_index.cpp",
        "asu_hbm_index_lookup/asu_hbm_index_lookup_torch_adpt.h",
        "asu_hbm_index_lookup/op_host/CMakeLists.txt",
        "asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_def.cpp",
        "asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_proto.cpp",
        "asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_tiling.cpp",
        "asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_tiling.h",
        "asu_hbm_index_lookup/op_kernel/asu_hbm_index_lookup.cpp",
        "asu_hbm_index_maintain/asu_hbm_index_maintain_torch_adpt.h",
        "asu_hbm_index_maintain/op_host/CMakeLists.txt",
        "asu_hbm_index_maintain/op_host/asu_hbm_index_maintain_def.cpp",
        "asu_hbm_index_maintain/op_host/asu_hbm_index_maintain_proto.cpp",
        "asu_hbm_index_maintain/op_host/asu_hbm_index_maintain_tiling.cpp",
        "asu_hbm_index_maintain/op_host/asu_hbm_index_maintain_tiling.h",
        "asu_hbm_index_maintain/op_kernel/asu_hbm_index_maintain.cpp",
    ]

    for file_name in required_files:
        assert (ASCEND_OPS_DIR / file_name).is_file(), file_name


def test_root_cmake_discovers_both_ops_without_original_ops_dependency():
    cmake = read("CMakeLists.txt")

    assert "asu_hbm_index_lookup" in cmake
    assert "asu_hbm_index_maintain" in cmake
    assert "../ops" not in cmake
    assert "ASU-Ascend/ops" not in cmake


def test_torch_binding_fragment_registers_public_ops():
    binding = read("torch_binding_asu_hbm_index.cpp")

    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in binding
    assert "asu_hbm_index_lookup" in binding
    assert "asu_hbm_index_maintain" in binding
    assert "torch::kPrivateUse1" in binding


def test_lookup_operator_keeps_original_shape_contract_and_vllm_symbols():
    kernel = read("asu_hbm_index_lookup/op_kernel/asu_hbm_index_lookup.cpp")
    host_def = read("asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_def.cpp")
    adapter = read("asu_hbm_index_lookup/asu_hbm_index_lookup_torch_adpt.h")

    for token in [
        "INDEX_SIZE = 128U * 1024U",
        "SLOT_COUNT = 10U * 1024U",
        "FREE_SLOT_COUNT = 2U * 1024U",
        "QUERY_COUNT = 2U * 1024U",
        "INDEX_TILE_LEN = 16U * 1024U",
        "NOT_FOUND = -1",
    ]:
        assert token in kernel

    assert 'extern "C" __global__ __aicore__ void asu_hbm_index_lookup' in kernel
    assert "class AsuHbmIndexLookup : public OpDef" in host_def
    assert "ge::DT_INT32" in host_def
    assert "aclnnAsuHbmIndexLookup" in adapter
    assert "slot_out" in adapter


def test_maintain_operator_keeps_original_eviction_contract_and_vllm_symbols():
    kernel = read("asu_hbm_index_maintain/op_kernel/asu_hbm_index_maintain.cpp")
    host_def = read("asu_hbm_index_maintain/op_host/asu_hbm_index_maintain_def.cpp")
    adapter = read("asu_hbm_index_maintain/asu_hbm_index_maintain_torch_adpt.h")

    for token in [
        "INDEX_SIZE = 128U * 1024U",
        "SLOT_COUNT = 10U * 1024U",
        "FREE_SLOT_COUNT = 2U * 1024U",
        "QUERY_COUNT = 2U * 1024U",
        "PROTECTED_WORD_BITS = 64U",
        "NOT_FOUND = -1",
        "Hash32",
        "IsProtectedSlot",
    ]:
        assert token in kernel

    assert 'extern "C" __global__ __aicore__ void asu_hbm_index_maintain' in kernel
    assert "class AsuHbmIndexMaintain : public OpDef" in host_def
    assert "ge::DT_INT32" in host_def
    assert "aclnnAsuHbmIndexMaintain" in adapter
    assert "seed" in adapter
