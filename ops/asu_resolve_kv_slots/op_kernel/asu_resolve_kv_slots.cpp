#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t HBM_RESIDENT = 2;
}  // namespace

class AsuResolveKvSlotsKernel {
public:
    __aicore__ inline AsuResolveKvSlotsKernel() {}

    __aicore__ inline void Init(
        GM_ADDR originalTopkIndices,
        GM_ADDR actualSeqLen,
        GM_ADDR managedPrefixLen,
        GM_ADDR tokenState,
        GM_ADDR asuRecordAddr,
        GM_ADDR hbmSlotOfToken,
        GM_ADDR slotOwnerToken,
        GM_ADDR freeSlotStack,
        GM_ADDR freeSlotCount,
        GM_ADDR originalBlockTable,
        GM_ADDR kvCache0,
        GM_ADDR kvCache1,
        GM_ADDR asuKvCache0,
        GM_ADDR asuKvCache1,
        GM_ADDR resolvedKvSlots,
        const AsuResolveKvSlotsTilingData* tilingData)
    {
        totalTopk_ = tilingData->totalTopk;
        blockSize_ = tilingData->blockSize;
        kv0BytesPerSlot_ = tilingData->kv0BytesPerSlot;
        kv1BytesPerSlot_ = tilingData->kv1BytesPerSlot;

        originalTopkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(originalTopkIndices), totalTopk_);
        actualSeqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(actualSeqLen), 1);
        managedPrefixLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(managedPrefixLen), 1);
        tokenStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenState));
        asuRecordAddrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(asuRecordAddr));
        hbmSlotOfTokenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(hbmSlotOfToken));
        slotOwnerTokenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotOwnerToken));
        freeSlotStackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(freeSlotStack));
        freeSlotCountGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(freeSlotCount), 1);
        originalBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(originalBlockTable));
        kvCache0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache0));
        kvCache1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache1));
        asuKvCache0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(asuKvCache0));
        asuKvCache1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(asuKvCache1));
        resolvedSlotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(resolvedKvSlots), totalTopk_);
    }

    __aicore__ inline void Process()
    {
        int32_t actualSeqLen = actualSeqLenGm_.GetValue(0);
        int32_t managedPrefixLen = managedPrefixLenGm_.GetValue(0);
        (void)actualSeqLen;

        for (uint32_t i = 0; i < totalTopk_; ++i) {
            int32_t tokenId = originalTopkGm_.GetValue(i);
            int32_t slot = 0;

            if (tokenId >= managedPrefixLen) {
                slot = ResolveTailSlot(tokenId);
            } else {
                slot = ResolveManagedSlot(tokenId);
            }

            resolvedSlotsGm_.SetValue(i, slot);
        }
    }

private:
    __aicore__ inline int32_t ResolveTailSlot(int32_t tokenId)
    {
        int32_t logicalBlock = tokenId / static_cast<int32_t>(blockSize_);
        int32_t offset = tokenId - logicalBlock * static_cast<int32_t>(blockSize_);
        int32_t physicalBlock = originalBlockTableGm_.GetValue(static_cast<uint32_t>(logicalBlock));
        return physicalBlock * static_cast<int32_t>(blockSize_) + offset;
    }

    __aicore__ inline int32_t ResolveManagedSlot(int32_t tokenId)
    {
        int32_t state = tokenStateGm_.GetValue(static_cast<uint32_t>(tokenId));
        if (state == HBM_RESIDENT) {
            return hbmSlotOfTokenGm_.GetValue(static_cast<uint32_t>(tokenId));
        }

        int32_t freeCount = freeSlotCountGm_.GetValue(0);
        int32_t nextCount = freeCount - 1;
        int32_t slot = freeSlotStackGm_.GetValue(static_cast<uint32_t>(nextCount));
        freeSlotCountGm_.SetValue(0, nextCount);

        int32_t asuRecord = asuRecordAddrGm_.GetValue(static_cast<uint32_t>(tokenId));
        CopyBytes(kvCache0Gm_,
                  static_cast<uint32_t>(slot) * kv0BytesPerSlot_,
                  asuKvCache0Gm_,
                  static_cast<uint32_t>(asuRecord) * kv0BytesPerSlot_,
                  kv0BytesPerSlot_);
        CopyBytes(kvCache1Gm_,
                  static_cast<uint32_t>(slot) * kv1BytesPerSlot_,
                  asuKvCache1Gm_,
                  static_cast<uint32_t>(asuRecord) * kv1BytesPerSlot_,
                  kv1BytesPerSlot_);

        tokenStateGm_.SetValue(static_cast<uint32_t>(tokenId), HBM_RESIDENT);
        hbmSlotOfTokenGm_.SetValue(static_cast<uint32_t>(tokenId), slot);
        slotOwnerTokenGm_.SetValue(static_cast<uint32_t>(slot), tokenId);

        return slot;
    }

    __aicore__ inline void CopyBytes(
        GlobalTensor<uint8_t>& dst,
        uint32_t dstOffset,
        GlobalTensor<uint8_t>& src,
        uint32_t srcOffset,
        uint32_t byteCount)
    {
        for (uint32_t i = 0; i < byteCount; ++i) {
            uint8_t value = src.GetValue(srcOffset + i);
            dst.SetValue(dstOffset + i, value);
        }
    }

private:
    GlobalTensor<int32_t> originalTopkGm_;
    GlobalTensor<int32_t> actualSeqLenGm_;
    GlobalTensor<int32_t> managedPrefixLenGm_;
    GlobalTensor<int32_t> tokenStateGm_;
    GlobalTensor<int32_t> asuRecordAddrGm_;
    GlobalTensor<int32_t> hbmSlotOfTokenGm_;
    GlobalTensor<int32_t> slotOwnerTokenGm_;
    GlobalTensor<int32_t> freeSlotStackGm_;
    GlobalTensor<int32_t> freeSlotCountGm_;
    GlobalTensor<int32_t> originalBlockTableGm_;
    GlobalTensor<int32_t> resolvedSlotsGm_;

    GlobalTensor<uint8_t> kvCache0Gm_;
    GlobalTensor<uint8_t> kvCache1Gm_;
    GlobalTensor<uint8_t> asuKvCache0Gm_;
    GlobalTensor<uint8_t> asuKvCache1Gm_;

    uint32_t totalTopk_;
    uint32_t blockSize_;
    uint32_t kv0BytesPerSlot_;
    uint32_t kv1BytesPerSlot_;
};

extern "C" __global__ __aicore__ void asu_resolve_kv_slots(
    GM_ADDR originalTopkIndices,
    GM_ADDR actualSeqLen,
    GM_ADDR managedPrefixLen,
    GM_ADDR tokenState,
    GM_ADDR asuRecordAddr,
    GM_ADDR hbmSlotOfToken,
    GM_ADDR slotOwnerToken,
    GM_ADDR freeSlotStack,
    GM_ADDR freeSlotCount,
    GM_ADDR originalBlockTable,
    GM_ADDR kvCache0,
    GM_ADDR kvCache1,
    GM_ADDR asuKvCache0,
    GM_ADDR asuKvCache1,
    GM_ADDR resolvedKvSlots,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() != 0 || !TILING_KEY_IS(1)) {
        return;
    }

    AsuResolveKvSlotsKernel op;
    op.Init(originalTopkIndices, actualSeqLen, managedPrefixLen, tokenState,
            asuRecordAddr, hbmSlotOfToken, slotOwnerToken, freeSlotStack,
            freeSlotCount, originalBlockTable, kvCache0, kvCache1,
            asuKvCache0, asuKvCache1, resolvedKvSlots, &tilingData);
    op.Process();
}
