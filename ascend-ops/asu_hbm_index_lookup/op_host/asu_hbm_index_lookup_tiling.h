#ifndef ASU_HBM_INDEX_LOOKUP_TILING_H
#define ASU_HBM_INDEX_LOOKUP_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsuHbmIndexLookupTilingData)
TILING_DATA_FIELD_DEF(uint32_t, reqNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsuHbmIndexLookup, AsuHbmIndexLookupTilingData)
}  // namespace optiling

#endif  // ASU_HBM_INDEX_LOOKUP_TILING_H
