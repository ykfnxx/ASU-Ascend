#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t INDEX_SIZE = 128U * 1024U;
constexpr uint32_t SLOT_COUNT = 10U * 1024U;
constexpr uint32_t FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t QUERY_COUNT = 2U * 1024U;
constexpr uint32_t PROTECTED_WORD_BITS = 64U;
constexpr uint32_t PROTECTED_WORD_COUNT = (SLOT_COUNT + PROTECTED_WORD_BITS - 1U) / PROTECTED_WORD_BITS;
constexpr int32_t NOT_FOUND = -1;

__aicore__ inline uint32_t Hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

class KernelAsuHbmIndexMaintain {
public:
    __aicore__ inline KernelAsuHbmIndexMaintain() {}

    __aicore__ inline void Init(GM_ADDR index,
                                GM_ADDR slotToIndex,
                                GM_ADDR freeSlots,
                                GM_ADDR freeHead,
                                GM_ADDR lastQuerySlots,
                                GM_ADDR indexOut,
                                GM_ADDR slotToIndexOut,
                                GM_ADDR freeSlotsOut,
                                GM_ADDR freeHeadOut,
                                uint32_t reqNum,
                                uint32_t seed,
                                TPipe* pipe)
    {
        (void)indexOut;
        (void)slotToIndexOut;
        (void)freeSlotsOut;
        (void)freeHeadOut;
        pipe_ = pipe;
        reqNum_ = reqNum;
        seed_ = seed;

        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), reqNum_ * INDEX_SIZE);
        slotToIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotToIndex), reqNum_ * SLOT_COUNT);
        freeSlotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(freeSlots), reqNum_ * FREE_SLOT_COUNT);
        freeHeadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(freeHead), reqNum_);
        lastQuerySlotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lastQuerySlots), reqNum_ * QUERY_COUNT);

        pipe_->InitBuffer(protectedBuf_, PROTECTED_WORD_COUNT * sizeof(uint64_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        auto protectedSlots = protectedBuf_.Get<uint64_t>();
        protectedSlots.SetSize(PROTECTED_WORD_COUNT);

        for (uint32_t reqId = coreId; reqId < reqNum_; reqId += blockNum) {
            uint32_t indexReqBase = reqId * INDEX_SIZE;
            uint32_t slotReqBase = reqId * SLOT_COUNT;
            uint32_t freeReqBase = reqId * FREE_SLOT_COUNT;
            uint32_t queryReqBase = reqId * QUERY_COUNT;
            int32_t head = freeHeadGm_.GetValue(reqId);
            if (head == 0) {
                continue;
            }

            ClearProtectedSlots(protectedSlots);
            for (uint32_t i = 0; i < QUERY_COUNT; ++i) {
                int32_t slot = lastQuerySlotsGm_.GetValue(queryReqBase + i);
                MarkProtectedSlot(protectedSlots, slot);
            }

            uint32_t slot = Hash32(seed_ ^ reqId) % SLOT_COUNT;
            while (head > 0) {
                int32_t indexId = slotToIndexGm_.GetValue(slotReqBase + slot);
                if (indexId != NOT_FOUND && IsProtectedSlot(protectedSlots, slot) == 0) {
                    slotToIndexGm_.SetValue(slotReqBase + slot, NOT_FOUND);
                    indexGm_.SetValue(indexReqBase + static_cast<uint32_t>(indexId), NOT_FOUND);
                    --head;
                    freeSlotsGm_.SetValue(freeReqBase + static_cast<uint32_t>(head), static_cast<int32_t>(slot));
                }
                ++slot;
                if (slot == SLOT_COUNT) {
                    slot = 0;
                }
            }

            freeHeadGm_.SetValue(reqId, head);
        }
    }

private:
    __aicore__ inline void ClearProtectedSlots(LocalTensor<uint64_t>& protectedSlots)
    {
        for (uint32_t i = 0; i < PROTECTED_WORD_COUNT; ++i) {
            protectedSlots.SetValue(i, 0ULL);
        }
    }

    __aicore__ inline void MarkProtectedSlot(LocalTensor<uint64_t>& protectedSlots, int32_t slot)
    {
        uint32_t slotId = static_cast<uint32_t>(slot);
        uint32_t wordId = slotId / PROTECTED_WORD_BITS;
        uint32_t bitId = slotId % PROTECTED_WORD_BITS;
        uint64_t word = protectedSlots.GetValue(wordId);
        protectedSlots.SetValue(wordId, word | (1ULL << bitId));
    }

    __aicore__ inline int32_t IsProtectedSlot(LocalTensor<uint64_t>& protectedSlots, uint32_t slot)
    {
        uint64_t word = protectedSlots.GetValue(slot / PROTECTED_WORD_BITS);
        return (word & (1ULL << (slot % PROTECTED_WORD_BITS))) != 0ULL;
    }

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> protectedBuf_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int32_t> slotToIndexGm_;
    GlobalTensor<int32_t> freeSlotsGm_;
    GlobalTensor<int32_t> freeHeadGm_;
    GlobalTensor<int32_t> lastQuerySlotsGm_;
    uint32_t reqNum_;
    uint32_t seed_;
};

}  // namespace

extern "C" __global__ __aicore__ void asu_hbm_index_maintain(GM_ADDR index,
                                                              GM_ADDR slotToIndex,
                                                              GM_ADDR freeSlots,
                                                              GM_ADDR freeHead,
                                                              GM_ADDR lastQuerySlots,
                                                              GM_ADDR indexOut,
                                                              GM_ADDR slotToIndexOut,
                                                              GM_ADDR freeSlotsOut,
                                                              GM_ADDR freeHeadOut,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelAsuHbmIndexMaintain op;
    op.Init(index,
            slotToIndex,
            freeSlots,
            freeHead,
            lastQuerySlots,
            indexOut,
            slotToIndexOut,
            freeSlotsOut,
            freeHeadOut,
            tilingData.reqNum,
            tilingData.seed,
            &pipe);
    op.Process();
}
