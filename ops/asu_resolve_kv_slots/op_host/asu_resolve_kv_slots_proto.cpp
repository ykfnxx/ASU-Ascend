#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {

constexpr uint32_t ORIGINAL_TOPK_INDEX = 0;
constexpr uint32_t RESOLVED_SLOTS_INDEX = 0;

static ge::graphStatus InferShapeAsuResolveKvSlots(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuResolveKvSlots", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);

    const gert::Shape* topkShape = context->GetInputShape(ORIGINAL_TOPK_INDEX);
    OPS_LOG_E_IF_NULL(context, topkShape, return ge::GRAPH_FAILED);

    gert::Shape* outShape = context->GetOutputShape(RESOLVED_SLOTS_INDEX);
    OPS_LOG_E_IF_NULL(context, outShape, return ge::GRAPH_FAILED);

    outShape->SetDimNum(topkShape->GetDimNum());
    for (size_t i = 0; i < topkShape->GetDimNum(); ++i) {
        outShape->SetDim(i, topkShape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAsuResolveKvSlots(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuResolveKvSlots", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(RESOLVED_SLOTS_INDEX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuResolveKvSlots)
    .InferShape(InferShapeAsuResolveKvSlots)
    .InferDataType(InferDataTypeAsuResolveKvSlots);

}  // namespace ops
