#include "asu_hbm_index_lookup_simt_constants.h"

#include <cstdint>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

void CheckNpuContiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckNpuDtype(const at::Tensor& tensor,
                   at::ScalarType dtype,
                   const char* name)
{
    CheckNpuContiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == dtype,
                name, " has an invalid dtype: ", tensor.scalar_type());
}

void CheckSameDevice(const at::Tensor& tensor,
                     const c10::Device& expected_device,
                     const char* name)
{
    TORCH_CHECK(tensor.device() == expected_device,
                name, " must be on device ", expected_device,
                "; got ", tensor.device());
}

uint32_t CheckedReqNum(int64_t req_num)
{
    TORCH_CHECK(req_num > 0, "req_num must be positive");
    TORCH_CHECK(
        req_num <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "req_num exceeds uint32 range");
    return static_cast<uint32_t>(req_num);
}

void CheckShape2D(const at::Tensor& tensor,
                  int64_t rows,
                  int64_t columns,
                  const char* name)
{
    TORCH_CHECK(tensor.dim() == 2,
                name, " must be a 2D tensor; got ", tensor.dim(), " dimensions");
    TORCH_CHECK(tensor.size(0) == rows && tensor.size(1) == columns,
                name, " must have shape [", rows, ", ", columns,
                "]; got ", tensor.sizes());
}

int64_t WorkspaceElements(int64_t req_num)
{
    CheckedReqNum(req_num);
    TORCH_CHECK(
        static_cast<uint64_t>(req_num) <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                ASU_HBM_WORKSPACE_STRIDE,
        "workspace element count overflows int64");
    return static_cast<int64_t>(
        static_cast<uint64_t>(req_num) * ASU_HBM_WORKSPACE_STRIDE);
}

}  // namespace

int64_t asu_hbm_index_lookup_simt_workspace_size(int64_t req_num)
{
    return WorkspaceElements(req_num);
}

std::tuple<at::Tensor, at::Tensor> asu_hbm_index_lookup_simt(
    at::Tensor token_to_slot,
    at::Tensor slot_to_token,
    at::Tensor lru_slots,
    at::Tensor query_token_ids,
    int64_t req_num,
    at::Tensor workspace)
{
    CheckNpuDtype(token_to_slot, at::kInt, "token_to_slot");
    CheckNpuDtype(slot_to_token, at::kInt, "slot_to_token");
    CheckNpuDtype(lru_slots, at::kShort, "lru_slots");
    CheckNpuDtype(query_token_ids, at::kInt, "query_token_ids");

    const c10::Device device = token_to_slot.device();
    CheckSameDevice(slot_to_token, device, "slot_to_token");
    CheckSameDevice(lru_slots, device, "lru_slots");
    CheckSameDevice(query_token_ids, device, "query_token_ids");

    const uint32_t req_num_u32 = CheckedReqNum(req_num);
    CheckShape2D(token_to_slot, req_num, ASU_HBM_INDEX_SIZE,
                 "token_to_slot");
    CheckShape2D(slot_to_token, req_num, ASU_HBM_SLOT_COUNT,
                 "slot_to_token");
    CheckShape2D(lru_slots, req_num, ASU_HBM_SLOT_COUNT,
                 "lru_slots");
    CheckShape2D(query_token_ids, req_num, ASU_HBM_QUERY_COUNT,
                 "query_token_ids");

    const int64_t required_workspace = WorkspaceElements(req_num);
    if (!workspace.defined()) {
        workspace = at::empty({required_workspace}, query_token_ids.options());
    } else {
        CheckNpuDtype(workspace, at::kInt, "workspace");
        CheckSameDevice(workspace, device, "workspace");
        TORCH_CHECK(workspace.numel() >= required_workspace,
                    "workspace has ", workspace.numel(),
                    " int32 elements; need ", required_workspace);
    }

    at::Tensor slot_ids = at::empty_like(query_token_ids);
    at::Tensor miss_mask =
        at::empty(query_token_ids.sizes(),
                  query_token_ids.options().dtype(at::kBool));

    const c10_npu::OptionalNPUGuard npu_guard(device);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    asu_hbm_index_lookup_simt_do(
        acl_stream,
        reinterpret_cast<void*>(token_to_slot.data_ptr<int32_t>()),
        reinterpret_cast<void*>(slot_to_token.data_ptr<int32_t>()),
        reinterpret_cast<void*>(lru_slots.data_ptr<int16_t>()),
        const_cast<void*>(
            reinterpret_cast<const void*>(query_token_ids.data_ptr<int32_t>())),
        reinterpret_cast<void*>(slot_ids.data_ptr<int32_t>()),
        reinterpret_cast<void*>(miss_mask.data_ptr<bool>()),
        reinterpret_cast<void*>(workspace.data_ptr<int32_t>()),
        req_num_u32);

    return std::make_tuple(slot_ids, miss_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("workspace_size",
          &asu_hbm_index_lookup_simt_workspace_size,
          pybind11::arg("req_num"),
          "Return the int32 workspace element count.");
    m.def("asu_hbm_index_lookup_simt",
          &asu_hbm_index_lookup_simt,
          pybind11::arg("token_to_slot"),
          pybind11::arg("slot_to_token"),
          pybind11::arg("lru_slots"),
          pybind11::arg("query_token_ids"),
          pybind11::arg("req_num"),
          pybind11::arg("workspace") = at::Tensor(),
          "Ascend 950 SIMT token lookup, allocation, and approximate-LRU eviction.");
}
