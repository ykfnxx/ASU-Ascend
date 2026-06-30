#ifndef ASU_HBM_INDEX_LOOKUP_SIMT_CONSTANTS_H
#define ASU_HBM_INDEX_LOOKUP_SIMT_CONSTANTS_H

#include <cstdint>

constexpr uint32_t ASU_HBM_INDEX_SIZE = 128U * 1024U;
constexpr uint32_t ASU_HBM_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t ASU_HBM_FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t ASU_HBM_QUERY_COUNT = 2U * 1024U;
constexpr uint32_t ASU_HBM_SIMT_THREADS = 256U;
constexpr int32_t ASU_HBM_NOT_FOUND = -1;
constexpr int32_t ASU_HBM_CLAIMING = -2;

extern "C" void asu_hbm_index_lookup_simt_do(void* stream,
                                             void* index,
                                             void* slot_to_index,
                                             void* free_slots,
                                             void* alloc_count,
                                             void* query_index,
                                             void* slot_out,
                                             uint32_t req_num);

#endif
