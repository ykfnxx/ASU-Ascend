#include <algorithm>
#include <cstdint>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void hbm_lookup_vec_do(
    uint32_t blockDim,
    void* stream,
    void* tableKeys,
    void* tableStates,
    void* queryKeys,
    void* statesOut,
    uint32_t reqNum,
    uint32_t queryLen,
    int32_t notFound);

extern "C" void hbm_random_update_do(
    uint32_t blockDim,
    void* stream,
    void* tableKeys,
    void* tableStates,
    void* queryKeys,
    void* newStates,
    uint32_t reqNum,
    uint32_t queryLen,
    uint32_t seed,
    uint32_t updatePercent);

namespace {
constexpr int64_t INDEX_SIZE = 128 * 1024;
constexpr int64_t QUERY_TILE = 64;

struct LookupShape {
    int64_t reqNum;
    int64_t queryLen;
    bool singleReq;
};

void CheckInt32NpuContiguous(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(t.scalar_type() == at::kInt,
                name, " must be torch.int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

uint32_t ClampBlockDim(int64_t blockDim) {
    if (blockDim < 1) {
        return 1U;
    }
    if (blockDim > 64) {
        return 64U;
    }
    return static_cast<uint32_t>(blockDim);
}

int64_t AlignUpI64(int64_t x, int64_t align) {
    return ((x + align - 1) / align) * align;
}

LookupShape CheckLookupShapes(
    const at::Tensor& tableKeys,
    const at::Tensor& tableStates,
    const at::Tensor& queryKeys,
    const at::Tensor& newStates)
{
    TORCH_CHECK(tableKeys.dim() == 1 || tableKeys.dim() == 2,
                "table_keys must be 1-D [128K] or 2-D [req_num, 128K]");
    TORCH_CHECK(tableStates.sizes() == tableKeys.sizes(),
                "table_states must have the same shape as table_keys");
    TORCH_CHECK(newStates.sizes() == queryKeys.sizes(),
                "new_states must have the same shape as query_keys");

    bool singleReq = tableKeys.dim() == 1;
    int64_t reqNum = 1;
    int64_t queryLen = 0;
    if (singleReq) {
        TORCH_CHECK(tableKeys.numel() == INDEX_SIZE,
                    "table_keys must have exactly 128K elements");
        TORCH_CHECK(queryKeys.dim() == 1,
                    "query_keys must be 1-D when table_keys is 1-D");
        queryLen = queryKeys.numel();
    } else {
        TORCH_CHECK(tableKeys.size(1) == INDEX_SIZE,
                    "table_keys shape must be [req_num, 128K]");
        TORCH_CHECK(queryKeys.dim() == 2,
                    "query_keys must be 2-D [req_num, query_len] when table_keys is 2-D");
        TORCH_CHECK(queryKeys.size(0) == tableKeys.size(0),
                    "query_keys dim 0 must match table_keys dim 0");
        reqNum = tableKeys.size(0);
        queryLen = queryKeys.size(1);
    }

    TORCH_CHECK(reqNum >= 1, "req_num must be at least 1");
    TORCH_CHECK(reqNum <= static_cast<int64_t>(UINT32_MAX),
                "req_num is too large for this demo kernel");
    TORCH_CHECK(queryLen <= static_cast<int64_t>(UINT32_MAX),
                "query_len is too large for this demo kernel");

    int64_t paddedQueryLen = AlignUpI64(std::max<int64_t>(queryLen, 1), QUERY_TILE);
    int64_t queryTileNum = paddedQueryLen / QUERY_TILE;
    TORCH_CHECK(reqNum <= static_cast<int64_t>(UINT32_MAX) / queryTileNum,
                "req_num * ceil(query_len / 64) is too large for this demo kernel");
    TORCH_CHECK(reqNum <= static_cast<int64_t>(UINT32_MAX) / INDEX_SIZE,
                "req_num * 128K is too large for this demo kernel");
    if (queryLen > 0) {
        TORCH_CHECK(reqNum <= static_cast<int64_t>(UINT32_MAX) / queryLen,
                    "req_num * query_len is too large for this demo kernel");
        TORCH_CHECK(reqNum <= static_cast<int64_t>(UINT32_MAX) / paddedQueryLen,
                    "req_num * padded_query_len is too large for this demo kernel");
    }

    return {reqNum, queryLen, singleReq};
}
}  // namespace

at::Tensor lookup_random_update(
    const at::Tensor& tableKeys,
    at::Tensor& tableStates,
    const at::Tensor& queryKeys,
    const at::Tensor& newStates,
    int64_t seed,
    int64_t updatePercent = 5,
    int64_t blockDim = 8,
    int64_t notFound = -1,
    bool doUpdate = true)
{
    CheckInt32NpuContiguous(tableKeys, "table_keys");
    CheckInt32NpuContiguous(tableStates, "table_states");
    CheckInt32NpuContiguous(queryKeys, "query_keys");
    CheckInt32NpuContiguous(newStates, "new_states");

    LookupShape shape = CheckLookupShapes(tableKeys, tableStates, queryKeys, newStates);
    TORCH_CHECK(updatePercent >= 0 && updatePercent <= 100,
                "update_percent must be in [0, 100]");

    int64_t reqNum64 = shape.reqNum;
    int64_t queryLen64 = shape.queryLen;
    int64_t paddedLen64 = AlignUpI64(std::max<int64_t>(queryLen64, 1), QUERY_TILE);
    at::Tensor outFull = shape.singleReq
        ? at::empty({paddedLen64}, queryKeys.options())
        : at::empty({reqNum64, paddedLen64}, queryKeys.options());

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t bd = ClampBlockDim(blockDim);
    uint32_t updateBd = ClampBlockDim(std::min<int64_t>(blockDim, reqNum64));
    uint32_t reqNum = static_cast<uint32_t>(reqNum64);
    uint32_t qLen = static_cast<uint32_t>(queryLen64);
    uint32_t seed32 = static_cast<uint32_t>(seed);
    uint32_t updatePct = static_cast<uint32_t>(updatePercent);
    int32_t notFound32 = static_cast<int32_t>(notFound);

    if (qLen > 0U) {
        hbm_lookup_vec_do(
            bd,
            aclStream,
            const_cast<void*>(reinterpret_cast<const void*>(tableKeys.data_ptr<int32_t>())),
            reinterpret_cast<void*>(tableStates.data_ptr<int32_t>()),
            const_cast<void*>(reinterpret_cast<const void*>(queryKeys.data_ptr<int32_t>())),
            reinterpret_cast<void*>(outFull.data_ptr<int32_t>()),
            reqNum,
            qLen,
            notFound32);

        // Launch update after lookup on the same stream. This preserves the semantic:
        // states_out contains states before the random 5% update.
        if (doUpdate && updatePct > 0U) {
            hbm_random_update_do(
                updateBd,
                aclStream,
                const_cast<void*>(reinterpret_cast<const void*>(tableKeys.data_ptr<int32_t>())),
                reinterpret_cast<void*>(tableStates.data_ptr<int32_t>()),
                const_cast<void*>(reinterpret_cast<const void*>(queryKeys.data_ptr<int32_t>())),
                const_cast<void*>(reinterpret_cast<const void*>(newStates.data_ptr<int32_t>())),
                reqNum,
                qLen,
                seed32,
                updatePct);
        }
    }

    return shape.singleReq ? outFull.narrow(0, 0, queryLen64)
                           : outFull.narrow(1, 0, queryLen64);
}

at::Tensor lookup_only(
    const at::Tensor& tableKeys,
    const at::Tensor& tableStates,
    const at::Tensor& queryKeys,
    int64_t blockDim = 8,
    int64_t notFound = -1)
{
    auto dummyNewStates = at::empty_like(queryKeys);
    auto mutableStates = const_cast<at::Tensor&>(tableStates);
    return lookup_random_update(tableKeys, mutableStates, queryKeys, dummyNewStates,
                                /*seed=*/0, /*updatePercent=*/0, blockDim,
                                notFound, /*doUpdate=*/false);
}

void update_only(
    const at::Tensor& tableKeys,
    at::Tensor& tableStates,
    const at::Tensor& queryKeys,
    const at::Tensor& newStates,
    int64_t seed,
    int64_t updatePercent = 5,
    int64_t blockDim = 8)
{
    CheckInt32NpuContiguous(tableKeys, "table_keys");
    CheckInt32NpuContiguous(tableStates, "table_states");
    CheckInt32NpuContiguous(queryKeys, "query_keys");
    CheckInt32NpuContiguous(newStates, "new_states");

    LookupShape shape = CheckLookupShapes(tableKeys, tableStates, queryKeys, newStates);
    TORCH_CHECK(updatePercent >= 0 && updatePercent <= 100,
                "update_percent must be in [0, 100]");

    if (shape.queryLen == 0 || updatePercent == 0) {
        return;
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t updateBd = ClampBlockDim(std::min<int64_t>(blockDim, shape.reqNum));
    hbm_random_update_do(
        updateBd,
        aclStream,
        const_cast<void*>(reinterpret_cast<const void*>(tableKeys.data_ptr<int32_t>())),
        reinterpret_cast<void*>(tableStates.data_ptr<int32_t>()),
        const_cast<void*>(reinterpret_cast<const void*>(queryKeys.data_ptr<int32_t>())),
        const_cast<void*>(reinterpret_cast<const void*>(newStates.data_ptr<int32_t>())),
        static_cast<uint32_t>(shape.reqNum),
        static_cast<uint32_t>(shape.queryLen),
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(updatePercent));
}

PYBIND11_MODULE(hbm_lookup_update, m) {
    m.doc() = "HBM resident token-index lookup + random update kernels for Ascend 910B";
    m.def("lookup_random_update", &lookup_random_update,
          pybind11::arg("table_keys"),
          pybind11::arg("table_states"),
          pybind11::arg("query_keys"),
          pybind11::arg("new_states"),
          pybind11::arg("seed"),
          pybind11::arg("update_percent") = 5,
          pybind11::arg("block_dim") = 8,
          pybind11::arg("not_found") = -1,
          pybind11::arg("do_update") = true,
          "Lookup valid indexer token ids in resident per-request 128K table_states, "
          "return pre-update states, then update update_percent% of queried keys. "
          "not_found is kept for ABI compatibility and is ignored by the current kernel.");
    m.def("lookup_only", &lookup_only,
          pybind11::arg("table_keys"),
          pybind11::arg("table_states"),
          pybind11::arg("query_keys"),
          pybind11::arg("block_dim") = 8,
          pybind11::arg("not_found") = -1,
          "Lookup valid token ids in table_states only, without state update.");
    m.def("update_only", &update_only,
          pybind11::arg("table_keys"),
          pybind11::arg("table_states"),
          pybind11::arg("query_keys"),
          pybind11::arg("new_states"),
          pybind11::arg("seed"),
          pybind11::arg("update_percent") = 5,
          pybind11::arg("block_dim") = 8,
          "Update resident token-index table_states in place without running the lookup kernel.");
}
