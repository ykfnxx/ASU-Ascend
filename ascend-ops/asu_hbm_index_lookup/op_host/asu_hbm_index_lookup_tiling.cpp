#include "asu_hbm_index_lookup_tiling.h"

#include <algorithm>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t ATTR_REQ_NUM = 0U;
}  // namespace

static ge::graphStatus AsuHbmIndexLookupTilingFunc(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuHbmIndexLookup", "TilingContext is nullptr."),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* req_num_attr = attrs->GetAttrPointer<int64_t>(ATTR_REQ_NUM);
    OPS_LOG_E_IF_NULL(context, req_num_attr, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*req_num_attr <= 0, OPS_LOG_E(context->GetNodeName(), "req_num must be greater than 0."),
               return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0, OPS_LOG_E(context->GetNodeName(), "AIV core count is 0."),
               return ge::GRAPH_FAILED);

    AsuHbmIndexLookupTilingData tiling;
    uint32_t req_num = static_cast<uint32_t>(*req_num_attr);
    tiling.set_reqNum(req_num);
    context->SetBlockDim(std::min(req_num, aiv_num));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct AsuHbmIndexLookupCompileInfo {};

static ge::graphStatus TilingParseForAsuHbmIndexLookup(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsuHbmIndexLookup)
    .Tiling(AsuHbmIndexLookupTilingFunc)
    .TilingParse<AsuHbmIndexLookupCompileInfo>(TilingParseForAsuHbmIndexLookup);
}  // namespace optiling
