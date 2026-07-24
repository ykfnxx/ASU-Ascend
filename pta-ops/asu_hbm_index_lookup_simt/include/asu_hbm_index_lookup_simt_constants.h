#ifndef ASU_HBM_INDEX_LOOKUP_SIMT_CONSTANTS_H
#define ASU_HBM_INDEX_LOOKUP_SIMT_CONSTANTS_H

#include <cstdint>

constexpr uint32_t ASU_HBM_INDEX_SIZE = 128U * 1024U;
constexpr uint32_t ASU_HBM_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t ASU_HBM_QUERY_COUNT = 2U * 1024U;
constexpr uint32_t ASU_HBM_SIMT_THREADS = 256U;
constexpr int32_t ASU_HBM_NOT_FOUND = -1;
constexpr int32_t ASU_HBM_CLAIMING = -2;

// Per-request int32 workspace:
//   hit flags, compacted hit slots, compacted evictable slots,
//   per-thread hit/evict/miss counts, and four scalar counters.
constexpr uint64_t ASU_HBM_WORKSPACE_STRIDE =
    3ULL * ASU_HBM_SLOT_COUNT + 3ULL * ASU_HBM_SIMT_THREADS + 4ULL;

extern "C" void asu_hbm_index_lookup_simt_do(void* stream,
                                             void* token_to_slot,
                                             void* slot_to_token,
                                             void* lru_slots,
                                             void* query_token_ids,
                                             void* slot_ids,
                                             void* miss_mask,
                                             void* workspace,
                                             uint32_t req_num);

#endif
