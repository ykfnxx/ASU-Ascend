# KV Cache Offload v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the v0 SFA eager debug path described in `docs/v0/kv-cache-offload-v0-design.md`, with MicroKV-backed prefill persistence and a mock lookup/validation call before SFA.

**Architecture:** Keep MicroKV as an opaque byte store and put record serialization, bypass slot tables, miss loading, eviction bookkeeping, and comparison in `vllm_ascend/attention/offload_kv_cache_v0.py`. Wire vllm-ascend through narrow metadata and forward-context additions so the existing SFA path still calls the original SFA operator with original inputs.

**Tech Stack:** Python, PyTorch CPU/NPU tensors, vllm-ascend SFA eager path, MicroKV C++ daemon and Python client.

## Global Constraints

- Do not modify the original vLLM KV cache layout, `slot_mapping`, `block_table`, or SFA inputs.
- Do not implement the final lookup/SFA operator; the SFA insertion point calls a Python mock helper.
- Support eager debug only; graph capture must be rejected when the feature is enabled.
- MicroKV cache miss loads are synchronous and CPU-assisted.
- Store `k_nope` and `k_pe` together in one MicroKV record, then split into two bypass cache tensors at the same bypass slot.
- Keep bypass cache state isolated per `(req_id, layer_id)`.
- Keep `offload_slot_table[token_pos] -> offload_slot_id` fixed length, defaulting to 128K entries.
- Decode lookup validates only prefill-generated token positions.
- Do not implement CP, Sparse C8, indexer scale, quantized KV, or decode-token writeback in v0.

---

### Task 1: MicroKV MLA Token Namespace

**Files:**
- Modify: `MicroKV/python/microkv/client.py`
- Modify: `MicroKV/python/microkv/__init__.py`
- Modify: `MicroKV/tests/test_microkv_e2e.py`
- Modify: `MicroKV/docs/design.md`

**Interfaces:**
- Produces: `KV_MLA_TOKEN = 0`, exported by `microkv`.
- Confirms: C++ KV store already supports opaque bytes and per-type namespace.

- [ ] **Step 1: Write the failing test**

Add an e2e test that imports `KV_MLA_TOKEN`, stores a single opaque MLA record value using `make_key(..., cache_type=KV_MLA_TOKEN)`, and asserts `batch_get(KV_MLA_TOKEN, [key])` returns the same bytes.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python -m pytest tests/test_microkv_e2e.py::MicroKVE2ETest::test_mla_token_record_roundtrip_uses_opaque_value -q`
Expected: FAIL because `KV_MLA_TOKEN` is not exported.

- [ ] **Step 3: Implement minimal MicroKV adaptation**

Add `KV_MLA_TOKEN = KV_ATTENTION_K` in `client.py`, export it from `__init__.py`, and update docs to state that v0 type 0 stores a full MLA token record.

- [ ] **Step 4: Verify MicroKV tests**

Run: `make test`
Expected: all MicroKV tests pass.

### Task 2: vllm-ascend CPU-Testable v0 Helper

**Files:**
- Create: `vllm_ascend/attention/offload_kv_cache_v0.py`
- Create: `tests/ut/attention/test_offload_kv_cache_v0.py`

**Interfaces:**
- Produces: `pack_mla_token_record(k_nope, k_pe) -> bytes`.
- Produces: `unpack_mla_token_record(record, expected_k_nope_shape=None, expected_k_pe_shape=None, expected_dtype=None, device=None) -> tuple[torch.Tensor, torch.Tensor]`.
- Produces: `OffloadKVCacheV0Manager.persist_prefill_kv_to_microkv(layer_name, kv_cache, slot_mapping, attn_metadata)`.
- Produces: `OffloadKVCacheV0Manager.mock_lookup_and_validate(layer_name, kv_cache, topk_indices, attn_metadata)`.

- [ ] **Step 1: Write failing helper tests**

Cover record roundtrip, dtype mismatch, per `(req_id, layer_id)` slot isolation, eviction table invalidation, MicroKV miss skip, and mismatch detection with CPU tensors and a fake MicroKV client.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ut/attention/test_offload_kv_cache_v0.py -q`
Expected: FAIL because the helper module does not exist.

- [ ] **Step 3: Implement helper**

Implement record pack/unpack, layer ID parsing, fixed-length slot tables, synchronous fake/client-backed get/put calls, mock lookup, and comparison stats. Keep all tensor transfers explicit through `.cpu()` / `.to(device)`.

- [ ] **Step 4: Verify helper tests**

Run: `pytest tests/ut/attention/test_offload_kv_cache_v0.py -q`
Expected: all helper tests pass on CPU.

### Task 3: vllm-ascend Metadata and Forward Context Wiring

**Files:**
- Modify: `vllm_ascend/attention/utils.py`
- Modify: `vllm_ascend/attention/sfa_v1.py`
- Modify: `vllm_ascend/worker/model_runner_v1.py`
- Modify: `vllm_ascend/ascend_forward_context.py`
- Modify: `vllm_ascend/envs.py`

**Interfaces:**
- Produces: `AscendCommonAttentionMetadata.req_ids`, `token_req_indices_cpu`, `token_positions_cpu`, `prefill_lens_cpu`.
- Produces: `AscendSFAMetadata` carrying the same v0 CPU metadata.
- Produces: forward-context attribute `offload_kv_cache_v0`.
- Produces: env vars `VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE`, `MICROKV_SOCKET`, `VLLM_ASCEND_KV_OFFLOAD_V0_CAPACITY`.

- [ ] **Step 1: Write failing metadata tests**

Extend existing unit tests to assert the new metadata fields survive `AscendCommonAttentionMetadata.unpadded()` and SFA metadata builder output.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ut/attention/test_sfa_v1.py tests/ut/attention/test_attention_v1.py -q`
Expected: FAIL until fields are added and propagated.

- [ ] **Step 3: Implement metadata and context plumbing**

Add optional fields with defaults, copy them in `unpadded()`, propagate from SFA builder, capture request/token CPU arrays in `_prepare_inputs()`, construct the manager when the env var is enabled, and pass it through `set_ascend_forward_context()`.

- [ ] **Step 4: Verify metadata tests**

Run: `pytest tests/ut/attention/test_sfa_v1.py tests/ut/attention/test_attention_v1.py -q`
Expected: pass on CPU test environment.

### Task 4: SFA Mock Insertion Points

**Files:**
- Modify: `vllm_ascend/attention/sfa_v1.py`
- Modify: `tests/ut/attention/test_sfa_v1.py`

**Interfaces:**
- Consumes: `OffloadKVCacheV0Manager.persist_prefill_kv_to_microkv`.
- Consumes: `OffloadKVCacheV0Manager.mock_lookup_and_validate`.

- [ ] **Step 1: Write failing SFA wiring tests**

Use mocked manager methods to assert prefill persistence is called after native KV/indexer cache writes and mock lookup is called after `indexer_select_post_process()` with unchanged `topk_indices`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ut/attention/test_sfa_v1.py -q`
Expected: FAIL until SFA forward calls the manager.

- [ ] **Step 3: Implement minimal SFA insertion**

After `kv_cache[2]` scatter, call `persist_prefill_kv_to_microkv()` when a manager is present. After `indexer_select_post_process()`, call `mock_lookup_and_validate()` before `_execute_sparse_flash_attention_process()`. Reject CP, Sparse C8, and graph capture through helper guard.

- [ ] **Step 4: Verify SFA tests**

Run: `pytest tests/ut/attention/test_sfa_v1.py -q`
Expected: pass in CPU unit mode with mocks.

### Task 5: Final Verification

**Files:**
- All touched files.

- [ ] **Step 1: Run targeted MicroKV verification**

Run: `cd /home/solidyang/workspace/ASU-Ascend/MicroKV && make test`

- [ ] **Step 2: Run targeted vllm-ascend verification**

Run: `cd /home/solidyang/workspace/vllm-ascend && pytest tests/ut/attention/test_offload_kv_cache_v0.py tests/ut/attention/test_sfa_v1.py tests/ut/attention/test_attention_v1.py -q`

- [ ] **Step 3: Check formatting and accidental changes**

Run: `git diff --check` in both repositories and inspect `git status --short`.
