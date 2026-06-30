#include "asu_hbm_index_lookup_simt_constants.h"

#include <cstdint>
#include <limits>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

void CheckInt32NpuContiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                name, " must be torch.int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t RequiredElements(int64_t req_num, uint32_t per_req, const char* name)
{
    TORCH_CHECK(req_num > 0, "req_num must be positive");
    TORCH_CHECK(req_num <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "req_num is too large for ", name);
    TORCH_CHECK(req_num <= std::numeric_limits<int64_t>::max() / static_cast<int64_t>(per_req),
                name, " element count overflows int64");
    return req_num * static_cast<int64_t>(per_req);
}

void CheckMinElements(const at::Tensor& tensor, int64_t expected, const char* name)
{
    TORCH_CHECK(tensor.numel() >= expected,
                name, " must have at least ", expected, " elements; got ", tensor.numel());
}

void CheckExactElements(const at::Tensor& tensor, int64_t expected, const char* name)
{
    TORCH_CHECK(tensor.numel() == expected,
                name, " must have exactly ", expected, " elements; got ", tensor.numel());
}

void CheckSameDevice(const at::Tensor& tensor, const c10::Device& expected_device, const char* name)
{
    TORCH_CHECK(tensor.device() == expected_device,
                name, " must be on device ", expected_device, "; got ", tensor.device());
}

}  // namespace

at::Tensor asu_hbm_index_lookup_simt(at::Tensor index,
                                     at::Tensor slot_to_index,
                                     at::Tensor free_slots,
                                     at::Tensor query_index,
                                     int64_t req_num)
{
    CheckInt32NpuContiguous(index, "index");
    CheckInt32NpuContiguous(slot_to_index, "slot_to_index");
    CheckInt32NpuContiguous(free_slots, "free_slots");
    CheckInt32NpuContiguous(query_index, "query_index");

    const c10::Device device = index.device();
    CheckSameDevice(slot_to_index, device, "slot_to_index");
    CheckSameDevice(free_slots, device, "free_slots");
    CheckSameDevice(query_index, device, "query_index");

    const int64_t required_index = RequiredElements(req_num, ASU_HBM_INDEX_SIZE, "index");
    const int64_t required_slot_to_index = RequiredElements(req_num, ASU_HBM_SLOT_COUNT, "slot_to_index");
    const int64_t required_free_slots = RequiredElements(req_num, ASU_HBM_FREE_SLOT_COUNT, "free_slots");
    const int64_t required_query = RequiredElements(req_num, ASU_HBM_QUERY_COUNT, "query_index");

    CheckMinElements(index, required_index, "index");
    CheckMinElements(slot_to_index, required_slot_to_index, "slot_to_index");
    CheckMinElements(free_slots, required_free_slots, "free_slots");
    CheckExactElements(query_index, required_query, "query_index");

    at::Tensor slot_out = at::empty_like(query_index);
    at::Tensor alloc_count = at::empty({req_num}, query_index.options());
    const c10_npu::OptionalNPUGuard npu_guard(device);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t req_num_u32 = static_cast<uint32_t>(req_num);

    asu_hbm_index_lookup_simt_do(
        acl_stream,
        reinterpret_cast<void*>(index.data_ptr<int32_t>()),
        reinterpret_cast<void*>(slot_to_index.data_ptr<int32_t>()),
        reinterpret_cast<void*>(free_slots.data_ptr<int32_t>()),
        reinterpret_cast<void*>(alloc_count.data_ptr<int32_t>()),
        const_cast<void*>(reinterpret_cast<const void*>(query_index.data_ptr<int32_t>())),
        reinterpret_cast<void*>(slot_out.data_ptr<int32_t>()),
        req_num_u32);

    return slot_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("asu_hbm_index_lookup_simt", &asu_hbm_index_lookup_simt,
          "ASU HBM index lookup with miss allocation, Ascend 950 SIMT PTA");
}
