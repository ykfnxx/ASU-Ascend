#include <stdint.h>

namespace {

constexpr uint32_t INDEX_SIZE = 128U * 1024U;
constexpr uint32_t SLOT_COUNT = 10U * 1024U;
constexpr uint32_t FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t QUERY_COUNT = 2U * 1024U;
constexpr int32_t NOT_FOUND = -1;

uint32_t Hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

int IsLastQuerySlot(const int32_t* lastQuerySlots, int32_t slot)
{
    for (uint32_t i = 0; i < QUERY_COUNT; ++i) {
        if (lastQuerySlots[i] == slot) {
            return 1;
        }
    }
    return 0;
}

void MaintainEviction(int32_t* index,
                      int32_t* slotToIndex,
                      int32_t* freeSlots,
                      int32_t* freeHead,
                      const int32_t* lastQuerySlots,
                      uint32_t reqNum,
                      uint32_t seed)
{
    for (uint32_t reqId = 0; reqId < reqNum; ++reqId) {
        uint32_t indexReqBase = reqId * INDEX_SIZE;
        uint32_t slotReqBase = reqId * SLOT_COUNT;
        uint32_t freeReqBase = reqId * FREE_SLOT_COUNT;
        uint32_t queryReqBase = reqId * QUERY_COUNT;
        int32_t head = freeHead[reqId];
        uint32_t start = Hash32(seed ^ reqId) % SLOT_COUNT;

        for (uint32_t offset = 0; head > 0; ++offset) {
            uint32_t slot = (start + offset) % SLOT_COUNT;
            int32_t indexId = slotToIndex[slotReqBase + slot];
            if (indexId != NOT_FOUND &&
                IsLastQuerySlot(lastQuerySlots + queryReqBase, static_cast<int32_t>(slot)) == 0) {
                slotToIndex[slotReqBase + slot] = NOT_FOUND;
                index[indexReqBase + static_cast<uint32_t>(indexId)] = NOT_FOUND;
                --head;
                freeSlots[freeReqBase + static_cast<uint32_t>(head)] = static_cast<int32_t>(slot);
            }
        }

        freeHead[reqId] = head;
    }
}

}  // namespace

extern "C" void asu_hbm_index_maintain(void* index,
                                        void* slotToIndex,
                                        void* freeSlots,
                                        void* freeHead,
                                        const void* lastQuerySlots,
                                        uint32_t reqNum,
                                        uint32_t seed)
{
    MaintainEviction(static_cast<int32_t*>(index),
                     static_cast<int32_t*>(slotToIndex),
                     static_cast<int32_t*>(freeSlots),
                     static_cast<int32_t*>(freeHead),
                     static_cast<const int32_t*>(lastQuerySlots),
                     reqNum,
                     seed);
}
