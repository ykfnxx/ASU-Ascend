#include "asu_hbm_index_maintain_tiling.h"

#include <algorithm>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t ATTR_REQ_NUM = 0U;
constexpr uint32_t ATTR_SEED = 1U;
}  // namespace

static ge::graphStatus AsuHbmIndexMaintainTilingFunc(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuHbmIndexMaintain", "TilingContext is nullptr."),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* req_num_attr = attrs->GetAttrPointer<int64_t>(ATTR_REQ_NUM);
    const int64_t* seed_attr = attrs->GetAttrPointer<int64_t>(ATTR_SEED);
    OPS_LOG_E_IF_NULL(context, req_num_attr, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, seed_attr, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*req_num_attr <= 0, OPS_LOG_E(context->GetNodeName(), "req_num must be greater than 0."),
               return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0, OPS_LOG_E(context->GetNodeName(), "AIV core count is 0."),
               return ge::GRAPH_FAILED);

    AsuHbmIndexMaintainTilingData tiling;
    uint32_t req_num = static_cast<uint32_t>(*req_num_attr);
    tiling.set_reqNum(req_num);
    tiling.set_seed(static_cast<uint32_t>(*seed_attr));
    context->SetBlockDim(std::min(req_num, aiv_num));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct AsuHbmIndexMaintainCompileInfo {};

static ge::graphStatus TilingParseForAsuHbmIndexMaintain(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsuHbmIndexMaintain)
    .Tiling(AsuHbmIndexMaintainTilingFunc)
    .TilingParse<AsuHbmIndexMaintainCompileInfo>(TilingParseForAsuHbmIndexMaintain);
}  // namespace optiling
