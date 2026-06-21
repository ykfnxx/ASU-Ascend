#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t STATE_OUTPUT_COUNT = 4U;

ge::graphStatus CopyInputShapeToOutput(gert::InferShapeContext* context, uint32_t input_index, uint32_t output_index)
{
    const gert::Shape* input_shape = context->GetInputShape(input_index);
    OPS_LOG_E_IF_NULL(context, input_shape, return ge::GRAPH_FAILED);
    gert::Shape* output_shape = context->GetOutputShape(output_index);
    OPS_LOG_E_IF_NULL(context, output_shape, return ge::GRAPH_FAILED);
    output_shape->SetDimNum(input_shape->GetDimNum());
    for (size_t dim = 0; dim < input_shape->GetDimNum(); ++dim) {
        output_shape->SetDim(dim, input_shape->GetDim(dim));
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus InferShapeAsuHbmIndexMaintain(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuHbmIndexMaintain", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < STATE_OUTPUT_COUNT; ++i) {
        if (CopyInputShapeToOutput(context, i, i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAsuHbmIndexMaintain(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuHbmIndexMaintain", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < STATE_OUTPUT_COUNT; ++i) {
        context->SetOutputDataType(i, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuHbmIndexMaintain)
    .InferShape(InferShapeAsuHbmIndexMaintain)
    .InferDataType(InferDataTypeAsuHbmIndexMaintain);
}  // namespace ops
