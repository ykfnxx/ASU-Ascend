#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t INDEX_SIZE = 128U * 1024U;
constexpr uint32_t QUERY_TILE = 64;      // output staging tile; wrapper allocates states_out padded to this.

__aicore__ inline uint32_t CeilDivU32(uint32_t x, uint32_t y) {
    return (x + y - 1U) / y;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t x, uint32_t align) {
    return CeilDivU32(x, align) * align;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

__aicore__ inline uint32_t Hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        uint32_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

__aicore__ inline uint32_t PickCoprimeA(uint32_t seed, uint32_t n)
{
    if (n <= 1U) {
        return 1U;
    }
    uint32_t a = Hash32(seed) % n;
    if (a == 0U) {
        a = 1U;
    }
    while (GcdU32(a, n) != 1U) {
        ++a;
        if (a >= n) {
            a = 1U;
        }
    }
    return a;
}

class KernelHbmLookupVec {
public:
    __aicore__ inline KernelHbmLookupVec() {}

    __aicore__ inline void Init(
        GM_ADDR tableKeys,
        GM_ADDR tableStates,
        GM_ADDR queryKeys,
        GM_ADDR statesOut,
        uint32_t reqNum,
        uint32_t queryLen,
        int32_t notFound,
        TPipe* pipe)
    {
        reqNum_ = reqNum;
        queryLen_ = queryLen;
        paddedQueryLen_ = AlignUpU32(queryLen_, QUERY_TILE);
        pipe_ = pipe;

        (void)notFound;
        (void)tableKeys;
        tableStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableStates), reqNum_ * INDEX_SIZE);
        queryKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryKeys), reqNum_ * queryLen_);
        statesOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(statesOut), reqNum_ * paddedQueryLen_);

        pipe_->InitBuffer(outTileBuf_, QUERY_TILE * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t queryTileNum = CeilDivU32(queryLen_, QUERY_TILE);
        uint32_t totalTileNum = reqNum_ * queryTileNum;

        auto outTile = outTileBuf_.Get<int32_t>();
        outTile.SetSize(QUERY_TILE);

        for (uint32_t tileId = coreId; tileId < totalTileNum; tileId += blockNum) {
            uint32_t reqId = tileId / queryTileNum;
            uint32_t reqTileId = tileId - reqId * queryTileNum;
            uint32_t qBase = reqTileId * QUERY_TILE;
            uint32_t valid = MinU32(QUERY_TILE, queryLen_ - qBase);
            uint32_t indexBase = reqId * INDEX_SIZE;
            uint32_t queryBase = reqId * queryLen_;
            uint32_t outBase = reqId * paddedQueryLen_;

            for (uint32_t i = 0; i < QUERY_TILE; ++i) {
                int32_t outVal = 0;
                if (i < valid) {
                    uint32_t key = static_cast<uint32_t>(
                        queryKeysGm_.GetValue(queryBase + qBase + i));
                    outVal = tableStatesGm_.GetValue(indexBase + key);
                }
                outTile.SetValue(i, outVal);
            }

            PipeBarrier<PIPE_ALL>();
            // statesOut is allocated padded to QUERY_TILE in the pybind wrapper.
            // DataCopy avoids multi-core GlobalTensor::SetValue DCache/cacheline hazards.
            DataCopy(statesOutGm_[outBase + qBase], outTile, QUERY_TILE);
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECOUT> outTileBuf_;

    GlobalTensor<int32_t> tableStatesGm_;
    GlobalTensor<int32_t> queryKeysGm_;
    GlobalTensor<int32_t> statesOutGm_;

    uint32_t reqNum_;
    uint32_t queryLen_;
    uint32_t paddedQueryLen_;
};

class KernelHbmRandomUpdate {
public:
    __aicore__ inline KernelHbmRandomUpdate() {}

    __aicore__ inline void Init(
        GM_ADDR tableKeys,
        GM_ADDR tableStates,
        GM_ADDR queryKeys,
        GM_ADDR newStates,
        uint32_t reqNum,
        uint32_t queryLen,
        uint32_t seed,
        uint32_t updatePercent,
        TPipe* pipe)
    {
        reqNum_ = reqNum;
        queryLen_ = queryLen;
        seed_ = seed;
        updatePercent_ = updatePercent;
        pipe_ = pipe;

        (void)tableKeys;
        tableStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableStates), reqNum_ * INDEX_SIZE);
        queryKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryKeys), reqNum_ * queryLen_);
        newStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(newStates), reqNum_ * queryLen_);
    }

    __aicore__ inline void Process()
    {
        if (queryLen_ == 0U || updatePercent_ == 0U) {
            return;
        }

        uint32_t updateNum = (queryLen_ * updatePercent_) / 100U;
        if (updateNum > queryLen_) {
            updateNum = queryLen_;
        }
        if (updateNum == 0U) {
            return;
        }

        uint32_t coreId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        for (uint32_t reqId = coreId; reqId < reqNum_; reqId += blockNum) {
            uint32_t indexBase = reqId * INDEX_SIZE;
            uint32_t queryBase = reqId * queryLen_;

            uint32_t reqSeed = seed_ ^ Hash32(reqId);
            uint32_t a = PickCoprimeA(reqSeed ^ 0x9e3779b9U, queryLen_);
            uint32_t b = Hash32(reqSeed ^ 0x85ebca6bU) % queryLen_;

            for (uint32_t t = 0; t < updateNum; ++t) {
                // Because a and queryLen are coprime, positions are unique for t in [0, queryLen).
                uint32_t pos = (static_cast<uint64_t>(a) * t + b) % queryLen_;
                int32_t key = queryKeysGm_.GetValue(queryBase + pos);
                int32_t newVal = newStatesGm_.GetValue(queryBase + pos);
                tableStatesGm_.SetValue(indexBase + static_cast<uint32_t>(key), newVal);
            }
        }
    }

private:
    TPipe* pipe_;

    GlobalTensor<int32_t> tableStatesGm_;
    GlobalTensor<int32_t> queryKeysGm_;
    GlobalTensor<int32_t> newStatesGm_;

    uint32_t reqNum_;
    uint32_t queryLen_;
    uint32_t seed_;
    uint32_t updatePercent_;
};
}  // namespace

extern "C" __global__ __aicore__ void hbm_lookup_vec(
    GM_ADDR tableKeys,
    GM_ADDR tableStates,
    GM_ADDR queryKeys,
    GM_ADDR statesOut,
    uint32_t reqNum,
    uint32_t queryLen,
    int32_t notFound)
{
    TPipe pipe;
    KernelHbmLookupVec op;
    op.Init(tableKeys, tableStates, queryKeys, statesOut, reqNum, queryLen, notFound, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void hbm_random_update(
    GM_ADDR tableKeys,
    GM_ADDR tableStates,
    GM_ADDR queryKeys,
    GM_ADDR newStates,
    uint32_t reqNum,
    uint32_t queryLen,
    uint32_t seed,
    uint32_t updatePercent)
{
    TPipe pipe;
    KernelHbmRandomUpdate op;
    op.Init(tableKeys, tableStates, queryKeys, newStates, reqNum, queryLen, seed, updatePercent, &pipe);
    op.Process();
}

// Host-callable wrappers. The pybind module links this library and calls these
// functions, avoiding dependency on generated aclrtlaunch_*.h include paths.
extern "C" void hbm_lookup_vec_do(
    uint32_t blockDim,
    void* stream,
    void* tableKeys,
    void* tableStates,
    void* queryKeys,
    void* statesOut,
    uint32_t reqNum,
    uint32_t queryLen,
    int32_t notFound)
{
#ifndef ASCENDC_CPU_DEBUG
    hbm_lookup_vec<<<blockDim, nullptr, stream>>>(tableKeys, tableStates, queryKeys,
                                                  statesOut, reqNum, queryLen, notFound);
#endif
}

extern "C" void hbm_random_update_do(
    uint32_t blockDim,
    void* stream,
    void* tableKeys,
    void* tableStates,
    void* queryKeys,
    void* newStates,
    uint32_t reqNum,
    uint32_t queryLen,
    uint32_t seed,
    uint32_t updatePercent)
{
#ifndef ASCENDC_CPU_DEBUG
    hbm_random_update<<<blockDim, nullptr, stream>>>(tableKeys, tableStates, queryKeys,
                                                     newStates, reqNum, queryLen, seed,
                                                     updatePercent);
#endif
}
