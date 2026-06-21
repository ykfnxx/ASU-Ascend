#include <stdint.h>

struct AsuHbmIndexMaintainArgs {
    uint64_t index;
    uint64_t slotToIndex;
    uint64_t freeSlots;
    uint64_t freeHead;
    uint64_t lastQuerySlots;
    uint32_t reqNum;
    uint32_t seed;
};

extern __global__ __aicpu__ uint32_t asu_hbm_index_maintain_kernel(void* args);

extern "C" void asu_hbm_index_maintain_do(uint32_t blockDim,
                                           void* stream,
                                           void* index,
                                           void* slotToIndex,
                                           void* freeSlots,
                                           void* freeHead,
                                           void* lastQuerySlots,
                                           uint32_t reqNum,
                                           uint32_t seed)
{
    (void)blockDim;

    AsuHbmIndexMaintainArgs args = {
        reinterpret_cast<uint64_t>(index),
        reinterpret_cast<uint64_t>(slotToIndex),
        reinterpret_cast<uint64_t>(freeSlots),
        reinterpret_cast<uint64_t>(freeHead),
        reinterpret_cast<uint64_t>(lastQuerySlots),
        reqNum,
        seed,
    };

    asu_hbm_index_maintain_kernel<<<1, nullptr, stream>>>(&args, sizeof(AsuHbmIndexMaintainArgs));
}
