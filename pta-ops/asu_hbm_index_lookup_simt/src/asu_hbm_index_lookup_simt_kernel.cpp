#include "asu_hbm_index_lookup_simt_constants.h"

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {

__simt_vf__ __launch_bounds__(ASU_HBM_SIMT_THREADS) inline void AsuHbmIndexLookupSimt(
    __gm__ int32_t* token_to_slot,
    __gm__ int32_t* slot_to_token,
    __gm__ int16_t* lru_slots,
    __gm__ int32_t* query_token_ids,
    __gm__ int32_t* slot_ids,
    __gm__ uint8_t* miss_mask,
    __gm__ int32_t* workspace,
    uint32_t req_id)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);

    __gm__ int32_t* req_token_to_slot =
        token_to_slot + static_cast<uint64_t>(req_id) * ASU_HBM_INDEX_SIZE;
    __gm__ int32_t* req_slot_to_token =
        slot_to_token + static_cast<uint64_t>(req_id) * ASU_HBM_SLOT_COUNT;
    __gm__ int16_t* req_lru =
        lru_slots + static_cast<uint64_t>(req_id) * ASU_HBM_SLOT_COUNT;
    __gm__ int32_t* req_query =
        query_token_ids + static_cast<uint64_t>(req_id) * ASU_HBM_QUERY_COUNT;
    __gm__ int32_t* req_slot_ids =
        slot_ids + static_cast<uint64_t>(req_id) * ASU_HBM_QUERY_COUNT;
    __gm__ uint8_t* req_miss_mask =
        miss_mask + static_cast<uint64_t>(req_id) * ASU_HBM_QUERY_COUNT;

    __gm__ int32_t* req_workspace =
        workspace + static_cast<uint64_t>(req_id) * ASU_HBM_WORKSPACE_STRIDE;
    __gm__ int32_t* hit_flags = req_workspace;
    __gm__ int32_t* hit_slots = hit_flags + ASU_HBM_SLOT_COUNT;
    __gm__ int32_t* evictable_slots = hit_slots + ASU_HBM_SLOT_COUNT;
    __gm__ int32_t* thread_hit_counts = evictable_slots + ASU_HBM_SLOT_COUNT;
    __gm__ int32_t* thread_evict_counts =
        thread_hit_counts + ASU_HBM_SIMT_THREADS;
    __gm__ int32_t* thread_miss_counts =
        thread_evict_counts + ASU_HBM_SIMT_THREADS;
    __gm__ int32_t* counters =
        thread_miss_counts + ASU_HBM_SIMT_THREADS;

    for (uint32_t slot = tid; slot < ASU_HBM_SLOT_COUNT; slot += thread_count) {
        hit_flags[slot] = 0;
    }
    if (tid < 4U) {
        counters[tid] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Direct token -> slot lookup. A unique miss claims token_to_slot with CAS.
    // Duplicate query tokens observe CLAIMING and do not request duplicate IO.
    for (uint32_t pos = tid; pos < ASU_HBM_QUERY_COUNT; pos += thread_count) {
        const int32_t token = req_query[pos];
        req_slot_ids[pos] = ASU_HBM_NOT_FOUND;
        req_miss_mask[pos] = 0U;
        if (token < 0 || token >= static_cast<int32_t>(ASU_HBM_INDEX_SIZE)) {
            continue;
        }

        __gm__ int32_t* token_slot = req_token_to_slot + static_cast<uint32_t>(token);
        const int32_t slot = *token_slot;
        if (slot >= 0) {
            req_slot_ids[pos] = slot;
            hit_flags[static_cast<uint32_t>(slot)] = 1;
        } else if (slot == ASU_HBM_NOT_FOUND) {
            const int32_t old =
                asc_atomic_cas(token_slot, ASU_HBM_NOT_FOUND, ASU_HBM_CLAIMING);
            if (old == ASU_HBM_NOT_FOUND) {
                req_miss_mask[pos] = 1U;
            } else if (old >= 0) {
                // Defensive support for an external writer completing the same
                // token before the CAS. Normal use forbids concurrent mutation.
                req_slot_ids[pos] = old;
                hit_flags[static_cast<uint32_t>(old)] = 1;
            }
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Stable partition of the prior LRU order. Slots hit by this query batch
    // move to the MRU suffix; every other slot is an eviction candidate.
    const uint32_t lru_chunk =
        (ASU_HBM_SLOT_COUNT + thread_count - 1U) / thread_count;
    const uint32_t lru_begin = tid * lru_chunk;
    uint32_t lru_end = lru_begin + lru_chunk;
    if (lru_end > ASU_HBM_SLOT_COUNT) {
        lru_end = ASU_HBM_SLOT_COUNT;
    }

    int32_t local_hits = 0;
    int32_t local_evictables = 0;
    for (uint32_t pos = lru_begin; pos < lru_end; ++pos) {
        const uint32_t slot = static_cast<uint32_t>(req_lru[pos]);
        if (hit_flags[slot] != 0) {
            ++local_hits;
        } else {
            ++local_evictables;
        }
    }
    thread_hit_counts[tid] = local_hits;
    thread_evict_counts[tid] = local_evictables;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t hit_prefix = 0;
    int32_t evict_prefix = 0;
    for (uint32_t i = 0; i < tid; ++i) {
        hit_prefix += thread_hit_counts[i];
        evict_prefix += thread_evict_counts[i];
    }
    int32_t hit_offset = hit_prefix;
    int32_t evict_offset = evict_prefix;
    for (uint32_t pos = lru_begin; pos < lru_end; ++pos) {
        const int32_t slot = static_cast<int32_t>(req_lru[pos]);
        if (hit_flags[static_cast<uint32_t>(slot)] != 0) {
            hit_slots[hit_offset++] = slot;
        } else {
            evictable_slots[evict_offset++] = slot;
        }
    }
    if (tid == thread_count - 1U) {
        counters[0] = hit_prefix + local_hits;
        counters[1] = evict_prefix + local_evictables;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Compact unique misses in query order. The nth miss reuses the nth
    // evictable slot, so allocation follows the prior approximate-LRU order.
    const uint32_t query_chunk =
        (ASU_HBM_QUERY_COUNT + thread_count - 1U) / thread_count;
    const uint32_t query_begin = tid * query_chunk;
    uint32_t query_end = query_begin + query_chunk;
    if (query_end > ASU_HBM_QUERY_COUNT) {
        query_end = ASU_HBM_QUERY_COUNT;
    }

    int32_t local_misses = 0;
    for (uint32_t pos = query_begin; pos < query_end; ++pos) {
        if (req_miss_mask[pos] != 0U) {
            ++local_misses;
        }
    }
    thread_miss_counts[tid] = local_misses;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t miss_prefix = 0;
    for (uint32_t i = 0; i < tid; ++i) {
        miss_prefix += thread_miss_counts[i];
    }
    int32_t miss_offset = miss_prefix;
    for (uint32_t pos = query_begin; pos < query_end; ++pos) {
        if (req_miss_mask[pos] == 0U) {
            continue;
        }

        const uint32_t miss_rank = static_cast<uint32_t>(miss_offset++);
        const uint32_t victim_slot =
            static_cast<uint32_t>(evictable_slots[miss_rank]);
        const int32_t victim_token = req_slot_to_token[victim_slot];
        if (victim_token >= 0 &&
            victim_token < static_cast<int32_t>(ASU_HBM_INDEX_SIZE)) {
            req_token_to_slot[static_cast<uint32_t>(victim_token)] =
                ASU_HBM_NOT_FOUND;
        }

        const int32_t token = req_query[pos];
        req_slot_to_token[victim_slot] = token;
        req_token_to_slot[static_cast<uint32_t>(token)] =
            static_cast<int32_t>(victim_slot);
        req_slot_ids[pos] = static_cast<int32_t>(victim_slot);
    }
    if (tid == thread_count - 1U) {
        counters[2] = miss_prefix + local_misses;
    }
    asc_threadfence_block();
    asc_syncthreads();

    const uint32_t evictable_count = static_cast<uint32_t>(counters[1]);
    const uint32_t miss_count = static_cast<uint32_t>(counters[2]);
    const uint32_t stale_count = evictable_count - miss_count;

    // HiSparse-style batch LRU approximation, from LRU to MRU:
    //   untouched stale slots + newly allocated miss slots + existing hit slots.
    for (uint32_t pos = tid; pos < ASU_HBM_SLOT_COUNT; pos += thread_count) {
        if (pos < stale_count) {
            req_lru[pos] = static_cast<int16_t>(
                evictable_slots[miss_count + pos]);
        } else if (pos < evictable_count) {
            req_lru[pos] = static_cast<int16_t>(
                evictable_slots[pos - stale_count]);
        } else {
            req_lru[pos] = static_cast<int16_t>(
                hit_slots[pos - evictable_count]);
        }
    }

    // Resolve duplicate misses after their canonical occurrence has installed
    // the final token -> slot mapping. Only the canonical occurrence keeps
    // miss_mask=1, so downstream IO is issued once per unique missing token.
    for (uint32_t pos = tid; pos < ASU_HBM_QUERY_COUNT; pos += thread_count) {
        const int32_t token = req_query[pos];
        if (req_slot_ids[pos] == ASU_HBM_NOT_FOUND && token >= 0 &&
            token < static_cast<int32_t>(ASU_HBM_INDEX_SIZE)) {
            req_slot_ids[pos] =
                req_token_to_slot[static_cast<uint32_t>(token)];
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void asu_hbm_index_lookup_simt_kernel(
    GM_ADDR token_to_slot,
    GM_ADDR slot_to_token,
    GM_ADDR lru_slots,
    GM_ADDR query_token_ids,
    GM_ADDR slot_ids,
    GM_ADDR miss_mask,
    GM_ADDR workspace,
    uint32_t req_num)
{
    const uint32_t req_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (req_id >= req_num) {
        return;
    }

    asc_vf_call<AsuHbmIndexLookupSimt>(
        dim3(ASU_HBM_SIMT_THREADS),
        reinterpret_cast<__gm__ int32_t*>(token_to_slot),
        reinterpret_cast<__gm__ int32_t*>(slot_to_token),
        reinterpret_cast<__gm__ int16_t*>(lru_slots),
        reinterpret_cast<__gm__ int32_t*>(query_token_ids),
        reinterpret_cast<__gm__ int32_t*>(slot_ids),
        reinterpret_cast<__gm__ uint8_t*>(miss_mask),
        reinterpret_cast<__gm__ int32_t*>(workspace),
        req_id);
}

extern "C" void asu_hbm_index_lookup_simt_do(void* stream,
                                             void* token_to_slot,
                                             void* slot_to_token,
                                             void* lru_slots,
                                             void* query_token_ids,
                                             void* slot_ids,
                                             void* miss_mask,
                                             void* workspace,
                                             uint32_t req_num)
{
#ifndef ASCENDC_CPU_DEBUG
    asu_hbm_index_lookup_simt_kernel<<<req_num, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(token_to_slot),
        reinterpret_cast<GM_ADDR>(slot_to_token),
        reinterpret_cast<GM_ADDR>(lru_slots),
        reinterpret_cast<GM_ADDR>(query_token_ids),
        reinterpret_cast<GM_ADDR>(slot_ids),
        reinterpret_cast<GM_ADDR>(miss_mask),
        reinterpret_cast<GM_ADDR>(workspace),
        req_num);
#endif
}
