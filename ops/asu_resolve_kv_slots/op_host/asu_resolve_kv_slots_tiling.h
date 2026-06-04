#ifndef ASU_RESOLVE_KV_SLOTS_TILING_H_
#define ASU_RESOLVE_KV_SLOTS_TILING_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AsuResolveKvSlotsTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTopk)
    TILING_DATA_FIELD_DEF(uint32_t, blockSize)
    TILING_DATA_FIELD_DEF(uint32_t, kv0BytesPerSlot)
    TILING_DATA_FIELD_DEF(uint32_t, kv1BytesPerSlot)
END_TILING_DATA_DEF;

struct AsuResolveKvSlotsCompileInfo {
    uint32_t totalCoreNum = 1;
};

REGISTER_TILING_DATA_CLASS(AsuResolveKvSlots, AsuResolveKvSlotsTilingData)

}  // namespace optiling

#endif  // ASU_RESOLVE_KV_SLOTS_TILING_H_
