#include "asu_hbm_index_lookup_simt_constants.h"

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {

__simt_vf__ __launch_bounds__(ASU_HBM_SIMT_THREADS) inline void AsuHbmIndexLookupSimt(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* alloc_count,
    __gm__ int32_t* query_index,
    __gm__ int32_t* slot_out,
    uint32_t req_id)
{
    const uint32_t query_base = req_id * ASU_HBM_QUERY_COUNT;
    __gm__ int32_t* req_index = index + req_id * ASU_HBM_INDEX_SIZE;
    __gm__ int32_t* req_slot_to_index = slot_to_index + req_id * ASU_HBM_SLOT_COUNT;
    __gm__ int32_t* req_free_slots = free_slots + req_id * ASU_HBM_FREE_SLOT_COUNT;
    __gm__ int32_t* req_query_index = query_index + query_base;
    __gm__ int32_t* req_slot_out = slot_out + query_base;

    const uint32_t thread_id = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    const uint32_t simt_block_id = static_cast<uint32_t>(blockIdx.x);
    const uint32_t linear_thread_id = thread_id + simt_block_id * thread_count;

    if (thread_id == 0U && simt_block_id == 0U) {
        alloc_count[req_id] = 0;
    }

    asc_threadfence_block();
    asc_syncthreads();

    for (uint32_t pos = linear_thread_id; pos < ASU_HBM_QUERY_COUNT; pos += thread_count) {
        const int32_t index_id = req_query_index[pos];
        __gm__ int32_t* slot_ptr = req_index + static_cast<uint32_t>(index_id);
        const int32_t slot = *slot_ptr;
        if (slot == ASU_HBM_NOT_FOUND) {
            const int32_t old_slot = asc_atomic_cas(slot_ptr, ASU_HBM_NOT_FOUND, ASU_HBM_CLAIMING);
            if (old_slot == ASU_HBM_NOT_FOUND) {
                const int32_t rank = asc_atomic_add(alloc_count + req_id, 1);
                const int32_t slot = req_free_slots[static_cast<uint32_t>(rank)];
                *slot_ptr = slot;
                req_slot_to_index[static_cast<uint32_t>(slot)] = index_id;
            }
        }
    }

    asc_threadfence_block();
    asc_syncthreads();

    for (uint32_t pos = linear_thread_id; pos < ASU_HBM_QUERY_COUNT; pos += thread_count) {
        const int32_t index_id = req_query_index[pos];
        req_slot_out[pos] = req_index[static_cast<uint32_t>(index_id)];
    }
}

}  // namespace

extern "C" __global__ __aicore__ void asu_hbm_index_lookup_simt_kernel(GM_ADDR index,
                                                                        GM_ADDR slot_to_index,
                                                                        GM_ADDR free_slots,
                                                                        GM_ADDR alloc_count,
                                                                        GM_ADDR query_index,
                                                                        GM_ADDR slot_out,
                                                                        uint32_t req_num)
{
    const uint32_t req_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (req_id >= req_num) {
        return;
    }

    asc_vf_call<AsuHbmIndexLookupSimt>(
        dim3(ASU_HBM_SIMT_THREADS),
        reinterpret_cast<__gm__ int32_t*>(index),
        reinterpret_cast<__gm__ int32_t*>(slot_to_index),
        reinterpret_cast<__gm__ int32_t*>(free_slots),
        reinterpret_cast<__gm__ int32_t*>(alloc_count),
        reinterpret_cast<__gm__ int32_t*>(query_index),
        reinterpret_cast<__gm__ int32_t*>(slot_out),
        req_id);
}

extern "C" void asu_hbm_index_lookup_simt_do(void* stream,
                                             void* index,
                                             void* slot_to_index,
                                             void* free_slots,
                                             void* alloc_count,
                                             void* query_index,
                                             void* slot_out,
                                             uint32_t req_num)
{
#ifndef ASCENDC_CPU_DEBUG
    asu_hbm_index_lookup_simt_kernel<<<req_num, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(index),
        reinterpret_cast<GM_ADDR>(slot_to_index),
        reinterpret_cast<GM_ADDR>(free_slots),
        reinterpret_cast<GM_ADDR>(alloc_count),
        reinterpret_cast<GM_ADDR>(query_index),
        reinterpret_cast<GM_ADDR>(slot_out),
        req_num);
#endif
}
