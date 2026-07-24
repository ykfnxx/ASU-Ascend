#ifndef ASU_LOOKUP_SIMT_TEST_KERNEL_OPERATOR_H
#define ASU_LOOKUP_SIMT_TEST_KERNEL_OPERATOR_H

#include <cstdint>

#define __aicore__
#define __global__
#define __gm__
#define __launch_bounds__(threads)
#define __simt_vf__

using GM_ADDR = void*;

struct dim3 {
    explicit dim3(uint32_t x_value) : x(x_value) {}
    uint32_t x;
};

struct AsuLookupTestBuiltinDim {
    uint32_t x = 0;
};

inline AsuLookupTestBuiltinDim threadIdx;
inline AsuLookupTestBuiltinDim blockDim{256};

namespace AscendC {
inline uint32_t GetBlockIdx()
{
    return 0;
}
}  // namespace AscendC

template <auto Function, typename... Args>
inline void asc_vf_call(dim3, Args&&...)
{
}

#endif
