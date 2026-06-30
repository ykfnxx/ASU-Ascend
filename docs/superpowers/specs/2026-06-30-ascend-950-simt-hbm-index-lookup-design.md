# Ascend 950 SIMT HBM Index Lookup PTA Design

Date: 2026-06-30

## Goal

Add a new Ascend 950 SIMT implementation of the ASU HBM index lookup operator in PTA form, without modifying the existing `ascend-ops/asu_hbm_index_lookup` or `ops/asu_hbm_index_lookup_aiv.cpp` sources.

The new operator keeps the current lookup-and-allocate semantics:

- Inputs: `index`, `slot_to_index`, `free_slots`, `free_head`, `query_index`, `req_num`.
- Output: `slot_out`, same shape as `query_index`.
- Side effects: update `index`, `slot_to_index`, and `free_head` in place when a query misses.
- Fixed layout constants:
  - `INDEX_SIZE = 128 * 1024`
  - `SLOT_COUNT = 10 * 1024`
  - `FREE_SLOT_COUNT = 2 * 1024`
  - `QUERY_COUNT = 2 * 1024`
  - `NOT_FOUND = -1`

## Chosen Approach

Use a standalone PTA/SIMT package under:

```text
ASU-Ascend/pta-ops/asu_hbm_index_lookup_simt/
```

The package will expose a PyTorch-callable wrapper and launch an Ascend C SIMT kernel directly on the current NPU stream. This keeps the new work isolated from the existing CANN custom-op package while matching the requested PTA style.

The current `ascend-ops` custom-op layout remains available for the existing AIV implementation and is not changed.

## References

- CANN asc-devkit SIMT programming model: `https://gitcode.com/cann/asc-devkit/tree/master/docs/guide/编程指南/编程模型/AI-Core-SIMT编程`
- CANN asc-devkit SIMT quickstart examples: `https://gitcode.com/cann/asc-devkit/tree/master/examples/03_simt_api/00_introduction/00_quickstart/hello_world_simt`
- CANN asc-devkit SIMT API index, including atomic APIs: `https://gitcode.com/cann/asc-devkit/blob/master/docs/api/README.md`
- Ascend custom operator invocation reference: `https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0049.html`
- Local SIMT-style examples: `vllm-ascend/csrc/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_v2_src_to_dst_op_simt.h`

## Architecture

The new package has three parts:

1. SIMT kernel source
   - Contains the Ascend C SIMT kernel.
   - Uses one thread block per request.
   - Uses multiple SIMT threads within the block to parallelize query lookup and final output.

2. PTA wrapper
   - Provides a PyTorch-facing function such as `asu_hbm_index_lookup_simt(...)`.
   - Validates tensor dtype, contiguity, device, shape, and `req_num`.
   - Allocates `slot_out = empty_like(query_index)`.
   - Launches the SIMT kernel on the current NPU stream.

3. Build and validation files
   - CMake/build script for the standalone PTA package.
   - Static tests to verify package layout and expected symbols.
   - Optional runtime validation script that compares against the existing Python reference when Ascend 950 and CANN 9.0 are available.

## Kernel Mapping

Launch shape:

```text
gridDim.x = req_num
blockDim.x = 256
```

Each thread block owns one request state:

```text
req_id = blockIdx.x
index_req_base = req_id * INDEX_SIZE
slot_req_base = req_id * SLOT_COUNT
free_req_base = req_id * FREE_SLOT_COUNT
query_req_base = req_id * QUERY_COUNT
```

Each SIMT thread processes query offsets by stride:

```text
for q = threadIdx.x; q < QUERY_COUNT; q += blockDim.x
```

## Miss Allocation Correctness

The existing AIV implementation behaves like a serial query loop per request during miss allocation. A naive SIMT implementation would be incorrect because duplicated missing query IDs could race and consume multiple free slots.

The SIMT design uses a three-phase flow with an internal sentinel:

```text
CLAIMING = -2
```

Phase 1: parallel claim

- Each thread reads `query_index[q]`.
- If `index[token] != NOT_FOUND`, no allocation is needed.
- If `index[token] == NOT_FOUND`, the thread attempts `atomicCAS(index[token], NOT_FOUND, CLAIMING)`.
- Only one thread can claim a unique missing token.

Phase 2: serial allocation within the request

- After a block-level barrier, `threadIdx.x == 0` scans `query_index[0..QUERY_COUNT)`.
- For each token whose `index[token] == CLAIMING`, it consumes the next free slot:

```text
slot = free_slots[free_head]
free_head += 1
index[token] = slot
slot_to_index[slot] = token
```

- Scanning query order preserves the existing per-request allocation order.
- Duplicate misses see the already assigned slot after the first allocation.

Phase 3: parallel output

- After another block-level barrier, all threads write:

```text
slot_out[q] = index[query_index[q]]
```

This gives SIMT parallelism for random lookup and output while keeping allocation deterministic and compatible with the existing contract.

## Preconditions

The wrapper enforces:

- All tensors are NPU tensors.
- All tensors are contiguous.
- All tensors have dtype `int32`.
- `req_num > 0`.
- `index.numel() >= req_num * INDEX_SIZE`.
- `slot_to_index.numel() >= req_num * SLOT_COUNT`.
- `free_slots.numel() >= req_num * FREE_SLOT_COUNT`.
- `free_head.numel() >= req_num`.
- `query_index.numel() >= req_num * QUERY_COUNT`.

The operator assumes:

- Every query ID is in `[0, INDEX_SIZE)`.
- The free pool has enough slots for all unique misses in each request.
- No other kernel mutates the same request state concurrently.

These assumptions match the current prototype contract.

## Error Handling

Host-side validation fails fast with `TORCH_CHECK` for dtype, device, contiguity, shape, and `req_num`.

Device-side invalid query IDs and free-list exhaustion are treated as contract violations. The first implementation will not add a separate error-output tensor or slow device-side validation pass because the target is a simple functional operator equivalent to the current implementation.

## Testing

Static tests:

- New files exist under `pta-ops/asu_hbm_index_lookup_simt`.
- The current `ascend-ops/asu_hbm_index_lookup` files are not modified.
- The SIMT source includes expected symbols and constants.
- The PTA wrapper exposes the expected PyTorch-callable function.

Reference tests:

- Reuse the existing Python reference behavior from `ops/scripts/asu_hbm_index_common.py`.
- Cover hit, miss, and mixed query patterns.
- Include duplicate missing query IDs to verify one allocation per unique token.
- Verify `slot_out`, `index`, `slot_to_index`, and `free_head`.

Runtime tests:

- Build and run only on an Ascend 950 environment with CANN 9.0/PTA support.
- Compare SIMT output against the reference for `req_num` values such as 1, 2, 8, and 16.
- Benchmark against the existing AIV implementation when both are available.

## Non-Goals

- Do not modify or replace the existing AIV CANN custom-op implementation.
- Do not redesign the HBM index layout.
- Do not add eviction or maintain logic.
- Do not make query length or index size dynamic in the first version.
- Do not fuse this operator into attention kernels.

## Open Decisions Resolved

- The operator includes miss allocation and free-slot updates.
- The implementation is added as a separate PTA/SIMT operator package.
- Duplicate miss correctness is handled with a `CLAIMING` sentinel plus request-local serial allocation.
- The first version targets Ascend 950 and CANN 9.0-style SIMT support.
