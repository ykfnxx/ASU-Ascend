#ifndef ASU_HBM_INDEX_MAINTAIN_TILING_H
#define ASU_HBM_INDEX_MAINTAIN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsuHbmIndexMaintainTilingData)
TILING_DATA_FIELD_DEF(uint32_t, reqNum);
TILING_DATA_FIELD_DEF(uint32_t, seed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsuHbmIndexMaintain, AsuHbmIndexMaintainTilingData)
}  // namespace optiling

#endif  // ASU_HBM_INDEX_MAINTAIN_TILING_H
