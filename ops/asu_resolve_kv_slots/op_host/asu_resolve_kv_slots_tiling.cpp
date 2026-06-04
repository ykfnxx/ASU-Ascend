#include "asu_resolve_kv_slots_tiling.h"
#include "log/ops_log.h"
#include "register/op_def_registry.h"

#include <cstdint>

namespace optiling {

constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 0;
constexpr uint32_t ATTR_KV0_BYTES_PER_SLOT_INDEX = 1;
constexpr uint32_t ATTR_KV1_BYTES_PER_SLOT_INDEX = 2;

static uint32_t ShapeNumel(const gert::StorageShape* shape)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() == 0) {
        return 0;
    }

    uint64_t numel = 1;
    const gert::Shape& storageShape = shape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        int64_t dim = storageShape.GetDim(i);
        if (dim <= 0) {
            return 0;
        }
        numel *= static_cast<uint64_t>(dim);
    }
    return static_cast<uint32_t>(numel);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    int32_t blockSize = *(attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX));
    int32_t kv0BytesPerSlot = *(attrs->GetAttrPointer<int32_t>(ATTR_KV0_BYTES_PER_SLOT_INDEX));
    int32_t kv1BytesPerSlot = *(attrs->GetAttrPointer<int32_t>(ATTR_KV1_BYTES_PER_SLOT_INDEX));
    uint32_t totalTopk = ShapeNumel(context->GetInputShape(0));

    AsuResolveKvSlotsTilingData tiling;
    tiling.set_totalTopk(totalTopk);
    tiling.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.set_kv0BytesPerSlot(static_cast<uint32_t>(kv0BytesPerSlot));
    tiling.set_kv1BytesPerSlot(static_cast<uint32_t>(kv1BytesPerSlot));

    context->SetTilingKey(1);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(1);

    OPS_LOG_I(context, "AsuResolveKvSlots totalTopk=%u blockSize=%d kv0Bytes=%d kv1Bytes=%d",
              totalTopk, blockSize, kv0BytesPerSlot, kv1BytesPerSlot);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AsuResolveKvSlots(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AsuResolveKvSlotsCompileInfo>();
    if (compileInfo != nullptr) {
        compileInfo->totalCoreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsuResolveKvSlots)
    .Tiling(TilingFunc)
    .TilingParse<AsuResolveKvSlotsCompileInfo>(TilingPrepare4AsuResolveKvSlots);

}  // namespace optiling
