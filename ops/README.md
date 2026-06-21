# ASU HBM Index Ops

This directory contains two prototype operator sources:

- `asu_hbm_index_lookup_aiv.cpp`: AIV lookup and miss allocation.
- `asu_hbm_index_maintain_aicpu.cpp`: AICPU index maintenance and eviction.

Each abstract req owns its own state:

```text
index          [reqNum, 128K]
slotToIndex    [reqNum, 10K]
freeSlots      [reqNum, 2K]
freeHead       [reqNum]
queryIndex     [reqNum, 2K]
slotOut        [reqNum, 2K]
lastQuerySlots [reqNum, 2K]
```

Build the AIV kernel source with CANN's AscendC kernel CMake:

```bash
bash build.sh lookup_aiv Ascend910B3
```

`lookup_aiv` uses CANN's `ascendc_kernel_cmake`; set
`ASCEND_CANN_PACKAGE_PATH` if the toolkit is not under
`/usr/local/Ascend/ascend-toolkit/latest`.

Build the direct AICPU dynamic library with Bisheng/Ascend C:

```bash
bash build.sh maintain_aicpu Ascend910B3
```

This produces `build/maintain_aicpu/lib/libasu_hbm_index_maintain_aicpu.so`.
Set `NPU_ARCH` if the default `dav-2201` is not correct for the target.

The old `msopgen` scaffold path is still available for framework packaging:

```bash
bash build.sh maintain_aicpu_msopgen
```

The `all` target keeps the two paths separate:

```bash
bash build.sh all Ascend910B3
```

## Validation and benchmark

The Python scripts under `scripts/` build deterministic index states and compare
operator output against the same greedy reference logic used by the prototypes.

Validate the AIV lookup kernel after building `lookup_aiv`:

```bash
python3 scripts/validate_hbm_index_ops.py --target lookup --req-num 2 --pattern mixed
```

Benchmark lookup latency and throughput:

```bash
python3 scripts/bench_hbm_index_ops.py --target lookup --req-num 1,2,4,8 --block-dim 1,2,4,8 --pattern hit
```

For miss or mixed allocation timing, reset the input state before every timed
iteration:

```bash
python3 scripts/bench_hbm_index_ops.py --target lookup --pattern mixed --reset-each-iter
```

The AICPU direct library is loaded through ctypes. After building
`maintain_aicpu`, no extra option is needed because the scripts search
`build/maintain_aicpu` by default:

```bash
python3 scripts/validate_hbm_index_ops.py --target maintain --strict-maintain
python3 scripts/bench_hbm_index_ops.py --target maintain
```

By default, maintain benchmarks use the `mixed` lookup state. That consumes
1022 of the 2048 free slots per req, so maintain refills about 49.9% of the
free pool. Use `--evict-ratio` to directly control this ratio for maintain
validation and benchmarks:

```bash
python3 scripts/validate_hbm_index_ops.py --target maintain --evict-ratio 0.25 --strict-maintain
python3 scripts/bench_hbm_index_ops.py --target maintain --evict-ratio 0.25
```

To measure a continuous maintain workload without per-iteration restore/copy
inside the timed range, expand the req dimension with `--chain-iters`. For
example, this builds an effective batch of `req_num * 64` req states, launches
one maintain op, and reports the average latency per logical `req_num` batch:

```bash
python3 scripts/bench_hbm_index_ops.py --target maintain \
  --req-num 2 \
  --evict-ratio 0.25 \
  --chain-iters 64
```

If the library is outside the default build tree, pass it explicitly:

```bash
python3 scripts/validate_hbm_index_ops.py --target maintain \
  --maintain-lib /path/to/libasu_hbm_index_maintain_aicpu.so \
  --strict-maintain
```

For framework-packaged AICPU operators visible through `torch.ops`, pass the
registered op name instead:

```bash
python3 scripts/validate_hbm_index_ops.py --target maintain --maintain-op _C_ascend.asu_hbm_index_maintain
```

If no maintain/index op appears, the AICPU package has not been built,
installed, or loaded into the current Python process yet. Check the build tree:

```bash
find build -type f \( -name "*.so" -o -name "*.run" -o -name "*.json" \) | sort
```

The scripts also still accept a Python wrapper:

```python
run_maintain(index, slot_to_index, free_slots, free_head, last_query_slots, req_num, seed)
```

It may update tensors in place or return
`(index, slot_to_index, free_slots, free_head)`.
