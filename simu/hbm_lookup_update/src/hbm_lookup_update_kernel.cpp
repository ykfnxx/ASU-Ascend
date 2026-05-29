#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t TABLE_SIZE = 2048;
constexpr uint32_t TABLE_TILE = 64;      // int32 compare: one vector iteration handles 64 elements on A2/A3.
constexpr uint32_t QUERY_TILE = 64;      // output staging tile; wrapper allocates states_out padded to this.
constexpr uint32_t REDUCE_WORK_SIZE = 128;
constexpr uint32_t REDUCE_OUT_SIZE = 8;
constexpr int32_t DEFAULT_NOT_FOUND = -1;
constexpr uint32_t INVALID_REQ = 0xffffffffU;

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

__aicore__ inline void FillQueryTile(LocalTensor<int32_t>& queryTile, int32_t qk)
{
    Duplicate<int32_t>(queryTile, qk, TABLE_TILE);
    PipeBarrier<PIPE_V>();
}

// Vector compare table_keys[tileBase:tileBase+TABLE_TILE] == qk, then use the bit mask
// only as a prefilter. We verify candidate bytes scalar-side to avoid depending on the
// bit-endianness of Compare's packed uint8 mask representation.
__aicore__ inline uint32_t FindKeyInTable(
    LocalTensor<int32_t>& tableKeysLocal,
    LocalTensor<int32_t>& queryTile,
    LocalTensor<uint8_t>& cmpMask,
    int32_t qk)
{
    FillQueryTile(queryTile, qk);

    for (uint32_t base = 0; base < TABLE_SIZE; base += TABLE_TILE) {
        Compare<int32_t, uint8_t>(cmpMask, tableKeysLocal[base], queryTile,
                                  CMPMODE::EQ, TABLE_TILE);
        PipeBarrier<PIPE_V>();

        // TABLE_TILE=64 => 8 bytes of packed mask.
        for (uint32_t byteIdx = 0; byteIdx < TABLE_TILE / 8U; ++byteIdx) {
            uint8_t packed = cmpMask.GetValue(byteIdx);
            if (packed == 0U) {
                continue;
            }

            // Do not rely on bit order; once a byte is nonzero, verify all 8 candidates.
            uint32_t localBegin = byteIdx * 8U;
            for (uint32_t k = 0; k < 8U; ++k) {
                uint32_t local = localBegin + k;
                int32_t tk = tableKeysLocal.GetValue(base + local);
                if (tk == qk) {
                    return base + local;
                }
            }
        }
    }
    return TABLE_SIZE;  // sentinel: not found
}

__aicore__ inline uint32_t FindKeyInTableVector(
    LocalTensor<int32_t>& tableKeysLocal,
    LocalTensor<uint8_t>& cmpMask,
    LocalTensor<float>& oneFlag,
    LocalTensor<float>& hitFlag,
    LocalTensor<float>& reduceOut,
    LocalTensor<float>& reduceWork,
    int32_t qk)
{
    CompareScalar<int32_t, uint8_t>(cmpMask, tableKeysLocal, qk, CMPMODE::EQ, TABLE_SIZE);
    PipeBarrier<PIPE_V>();
    Select<float, uint8_t>(hitFlag, cmpMask, oneFlag, 0.0f,
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, TABLE_SIZE);
    PipeBarrier<PIPE_V>();
    ReduceMax<float>(reduceOut, hitFlag, reduceWork, TABLE_SIZE, true);
    PipeBarrier<PIPE_V>();

    if (reduceOut.GetValue(0) == 0.0f) {
        return TABLE_SIZE;
    }
    return reduceOut.ReinterpretCast<uint32_t>().GetValue(1);
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
        notFound_ = notFound;
        pipe_ = pipe;

        tableKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableKeys), reqNum_ * TABLE_SIZE);
        tableStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableStates), reqNum_ * TABLE_SIZE);
        queryKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryKeys), reqNum_ * queryLen_);
        statesOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(statesOut), reqNum_ * paddedQueryLen_);

        pipe_->InitBuffer(tableKeysBuf_, TABLE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(tableStatesBuf_, TABLE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(cmpMaskBuf_, TABLE_SIZE * sizeof(uint8_t));
        pipe_->InitBuffer(oneFlagBuf_, TABLE_SIZE * sizeof(float));
        pipe_->InitBuffer(hitFlagBuf_, TABLE_SIZE * sizeof(float));
        pipe_->InitBuffer(reduceOutBuf_, REDUCE_OUT_SIZE * sizeof(float));
        pipe_->InitBuffer(reduceWorkBuf_, REDUCE_WORK_SIZE * sizeof(float));
        pipe_->InitBuffer(outTileBuf_, QUERY_TILE * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t queryTileNum = CeilDivU32(queryLen_, QUERY_TILE);
        uint32_t totalTileNum = reqNum_ * queryTileNum;

        auto tableKeysLocal = tableKeysBuf_.Get<int32_t>();
        auto tableStatesLocal = tableStatesBuf_.Get<int32_t>();
        auto cmpMask = cmpMaskBuf_.Get<uint8_t>();
        auto oneFlag = oneFlagBuf_.Get<float>();
        auto hitFlag = hitFlagBuf_.Get<float>();
        auto reduceOut = reduceOutBuf_.Get<float>();
        auto reduceWork = reduceWorkBuf_.Get<float>();
        auto outTile = outTileBuf_.Get<int32_t>();

        Duplicate<float>(oneFlag, 1.0f, TABLE_SIZE);
        PipeBarrier<PIPE_V>();
        uint32_t loadedReq = INVALID_REQ;

        for (uint32_t tileId = coreId; tileId < totalTileNum; tileId += blockNum) {
            uint32_t reqId = tileId / queryTileNum;
            uint32_t reqTileId = tileId - reqId * queryTileNum;
            uint32_t qBase = reqTileId * QUERY_TILE;
            uint32_t valid = MinU32(QUERY_TILE, queryLen_ - qBase);
            uint32_t tableBase = reqId * TABLE_SIZE;
            uint32_t queryBase = reqId * queryLen_;
            uint32_t outBase = reqId * paddedQueryLen_;

            if (reqId != loadedReq) {
                // Each AI Core keeps one req's 2K resident index and states in UB.
                DataCopy(tableKeysLocal, tableKeysGm_[tableBase], TABLE_SIZE);
                DataCopy(tableStatesLocal, tableStatesGm_[tableBase], TABLE_SIZE);
                PipeBarrier<PIPE_ALL>();
                loadedReq = reqId;
            }

            for (uint32_t i = 0; i < QUERY_TILE; ++i) {
                int32_t outVal = notFound_;
                if (i < valid) {
                    int32_t qk = queryKeysGm_.GetValue(queryBase + qBase + i);
                    uint32_t hit = FindKeyInTableVector(
                        tableKeysLocal, cmpMask, oneFlag, hitFlag, reduceOut, reduceWork, qk);
                    if (hit < TABLE_SIZE) {
                        outVal = tableStatesLocal.GetValue(hit);
                    }
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
    TBuf<TPosition::VECIN> tableKeysBuf_;
    TBuf<TPosition::VECIN> tableStatesBuf_;
    TBuf<TPosition::VECCALC> cmpMaskBuf_;
    TBuf<TPosition::VECCALC> oneFlagBuf_;
    TBuf<TPosition::VECCALC> hitFlagBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> reduceWorkBuf_;
    TBuf<TPosition::VECOUT> outTileBuf_;

    GlobalTensor<int32_t> tableKeysGm_;
    GlobalTensor<int32_t> tableStatesGm_;
    GlobalTensor<int32_t> queryKeysGm_;
    GlobalTensor<int32_t> statesOutGm_;

    uint32_t reqNum_;
    uint32_t queryLen_;
    uint32_t paddedQueryLen_;
    int32_t notFound_;
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

        tableKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableKeys), reqNum_ * TABLE_SIZE);
        tableStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableStates), reqNum_ * TABLE_SIZE);
        queryKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryKeys), reqNum_ * queryLen_);
        newStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(newStates), reqNum_ * queryLen_);

        pipe_->InitBuffer(tableKeysBuf_, TABLE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(tableStatesInBuf_, TABLE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(tableStatesCalcBuf_, TABLE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(tableStatesOutBuf_, TABLE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(queryTileBuf_, TABLE_TILE * sizeof(int32_t));
        pipe_->InitBuffer(cmpMaskBuf_, TABLE_TILE * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        if (queryLen_ == 0U || updatePercent_ == 0U) {
            return;
        }

        auto tableKeysLocal = tableKeysBuf_.Get<int32_t>();
        auto tableStatesInLocal = tableStatesInBuf_.Get<int32_t>();
        auto tableStatesCalcLocal = tableStatesCalcBuf_.Get<int32_t>();
        auto tableStatesOutLocal = tableStatesOutBuf_.Get<int32_t>();
        auto queryTile = queryTileBuf_.Get<int32_t>();
        auto cmpMask = cmpMaskBuf_.Get<uint8_t>();

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
            uint32_t tableBase = reqId * TABLE_SIZE;
            uint32_t queryBase = reqId * queryLen_;

            DataCopy(tableKeysLocal, tableKeysGm_[tableBase], TABLE_SIZE);
            DataCopy(tableStatesInLocal, tableStatesGm_[tableBase], TABLE_SIZE);
            PipeBarrier<PIPE_ALL>();
            DataCopy(tableStatesCalcLocal, tableStatesInLocal, TABLE_SIZE);
            PipeBarrier<PIPE_ALL>();

            uint32_t reqSeed = seed_ ^ Hash32(reqId);
            uint32_t a = PickCoprimeA(reqSeed ^ 0x9e3779b9U, queryLen_);
            uint32_t b = Hash32(reqSeed ^ 0x85ebca6bU) % queryLen_;

            for (uint32_t t = 0; t < updateNum; ++t) {
                // Because a and queryLen are coprime, positions are unique for t in [0, queryLen).
                uint32_t pos = (static_cast<uint64_t>(a) * t + b) % queryLen_;
                int32_t key = queryKeysGm_.GetValue(queryBase + pos);
                int32_t newVal = newStatesGm_.GetValue(queryBase + pos);

                uint32_t hit = FindKeyInTable(tableKeysLocal, queryTile, cmpMask, key);
                if (hit < TABLE_SIZE) {
                    tableStatesCalcLocal.SetValue(hit, newVal);
                }
            }

            PipeBarrier<PIPE_ALL>();
            // Write back the whole resident state table by DMA. The table is only 8KB.
            DataCopy(tableStatesOutLocal, tableStatesCalcLocal, TABLE_SIZE);
            PipeBarrier<PIPE_ALL>();
            DataCopy(tableStatesGm_[tableBase], tableStatesOutLocal, TABLE_SIZE);
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECIN> tableKeysBuf_;
    TBuf<TPosition::VECIN> tableStatesInBuf_;
    TBuf<TPosition::VECCALC> tableStatesCalcBuf_;
    TBuf<TPosition::VECOUT> tableStatesOutBuf_;
    TBuf<TPosition::VECCALC> queryTileBuf_;
    TBuf<TPosition::VECCALC> cmpMaskBuf_;

    GlobalTensor<int32_t> tableKeysGm_;
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
