# ASU Ascend Custom Ops

This directory follows the CANN custom-op layout used by vLLM-Ascend:

```text
ops/
  asu_resolve_kv_slots/
    op_host/
      *_def.cpp
      *_proto.cpp
      *_tiling.h
      *_tiling.cpp
      CMakeLists.txt
    op_kernel/
      *.cpp
```

The first functional operator is `AsuResolveKvSlots`. It implements the
single-request resolver from `docs/asu_g25_kvcache_functional_design.md`.

The implementation is intentionally simple:

- single req only;
- one AIV core;
- no performance optimization;
- no fallback path;
- no async ASU completion;
- ASU records are represented by input GM tensors `asu_kv_cache0` and
  `asu_kv_cache1` for functional testing.

State values used by the kernel:

```text
1 = ASU_ONLY
2 = HBM_RESIDENT
3 = TAIL_HBM
```

`resolved_kv_slots` keeps the same shape as `original_topk_indices`.
`sparse_indices` should still be passed to SFA unchanged as original token ids.
