# hbm_lookup_update

Ascend 910B / CANN KernelLaunch + PyBind demo for resident per-request HBM key/state tables.

## Semantics

The operator does **not** generate query keys internally. The caller passes `query_keys`.

Multi-request inputs:

- `table_keys`: `torch.int32`, NPU tensor, shape `[R, 2048]`, resident HBM index keys for `R` requests. Each row can be unordered.
- `table_states`: `torch.int32`, NPU tensor, shape `[R, 2048]`, resident HBM states. This tensor is updated in place.
- `query_keys`: `torch.int32`, NPU tensor, shape `[R, Q]`, external query keys.
- `new_states`: `torch.int32`, NPU tensor, shape `[R, Q]`, states used if a query position is selected for update.

The original single-request shapes `[2048]`, `[2048]`, `[Q]`, `[Q]` are still accepted and are treated as `R=1`.

Python call:

```python
states_out = hbm_lookup_update.lookup_random_update(
    table_keys, table_states, query_keys, new_states,
    seed=42, update_percent=5, block_dim=8, not_found=-1,
)
```

Separate lookup/update calls for profiling:

```python
states_out = hbm_lookup_update.lookup_only(
    table_keys, table_states, query_keys,
    block_dim=8, not_found=-1,
)

hbm_lookup_update.update_only(
    table_keys, table_states, query_keys, new_states,
    seed=42, update_percent=5, block_dim=8,
)
```

Meaning:

```cpp
// Kernel 1: multi-core lookup
for r in 0..R-1:
    for i in 0..Q-1:
        states_out[r, i] = table_states[r, j] if table_keys[r, j] == query_keys[r, i] else not_found

// Kernel 2: same stream, after lookup, per-request sequential update
for r in 0..R-1:
    for pos in random_unique_positions(floor(Q * update_percent / 100), seed, r):
        key = query_keys[r, pos]
        if table_keys[r, j] == key:
            table_states[r, j] = new_states[r, pos]
```

`states_out` returns **pre-update** states.

Assumptions:

- `table_keys` should be unique. If duplicates exist, the first matching slot in table order is used.
- Key and state dtype is `int32`.
- Table size is fixed at 2048 per request.

## Directory

```text
hbm_lookup_update/
├── CMakeLists.txt
├── cmake/
│   └── npu_lib.cmake
├── pybind/
│   └── pybind_hbm_lookup_update.cpp
├── run.sh
├── scripts/
│   ├── bench_lookup_update.py
│   ├── profile_lookup_update.py
│   └── test_lookup_update.py
└── src/
    └── hbm_lookup_update_kernel.cpp
```

## Build and test

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install pybind11
bash run.sh -v Ascend910B3
```

If your card reports a different SoC string, use that string:

```bash
npu-smi info
bash run.sh -v Ascend910B1
```

Skip test:

```bash
bash run.sh -v Ascend910B3 -t
```

Run benchmark after build:

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/bench_lookup_update.py \
  --mode lookup \
  --req-num 4 \
  --query-len 2048 \
  --block-dim 8 \
  --iters 100
```

Quick sweep without profiler:

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/bench_lookup_update.py \
  --mode lookup \
  --req-num 4,8,16 \
  --query-len 2048 \
  --block-dim 8,16,32,64 \
  --iters 50
```

`--mode lookup` runs lookup only, `--mode update` runs update only, and `--mode both` runs the combined `lookup_random_update` path.

## Profile lookup and update separately

`scripts/profile_lookup_update.py` can run the lookup kernel path and update kernel path independently. It delays importing `torch/torch_npu` until after argument parsing, so `--help` works on non-Ascend machines.

Show options:

```bash
python3 scripts/profile_lookup_update.py --help
```

Time lookup only:

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/profile_lookup_update.py \
  --mode lookup \
  --req-num 4 \
  --query-len 2048 \
  --block-dim 8 \
  --warmup 20 \
  --iters 200
```

Time update only:

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/profile_lookup_update.py \
  --mode update \
  --req-num 4 \
  --query-len 2048 \
  --update-percent 5 \
  --block-dim 8 \
  --warmup 20 \
  --iters 200
```

Collect torch_npu profiler traces for both paths:

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/profile_lookup_update.py \
  --mode both \
  --req-num 4 \
  --query-len 2048 \
  --update-percent 5 \
  --block-dim 8 \
  --warmup 20 \
  --iters 50 \
  --profile-dir ./profile_hbm_lookup_update \
  --profiler-level level2 \
  --aic-metrics pipe
```

When `--mode both` is used with `--profile-dir`, traces are written to separate `lookup/` and `update/` subdirectories. The script prints `host_ms_per_iter` and, when NPU events are available, `device_ms_per_iter`. Use `device_ms_per_iter` as the simple operator execution-time estimate.

For kernel-internal profile, inspect the files generated under `--profile-dir`. `kernel_details.csv` is the main file for this project: it contains NPU task name, `Duration(us)`, `Block Dim`, accelerator core type, and AI Core metric columns when `--aic-metrics` is enabled. `trace_view.json` can be opened with TensorBoard, Chrome tracing, Perfetto, or MindStudio for timeline inspection.

`--aic-metrics` selects one AI Core metric set per run: `pipe`, `arithmetic`, `memory`, `ub`, `l2cache`, or `resource`. For a quick simulation-level estimate, run once with `pipe`; rerun with `memory` or `ub` only if the first result suggests you need more detail.

## Design notes

The implementation uses two kernels under one Python function. This is deliberate: the lookup kernel is multi-core, while the update kernel runs after lookup on the same stream. The update kernel parallelizes across requests, but each request's resident state table is updated sequentially by one core. This avoids relying on cross-AI-Core synchronization inside one kernel and avoids multi-core random writes to the same request's HBM state table.

Lookup copies one request's 2K key/state table to UB and scans it with vector ops. For each query key, it runs `CompareScalar<int32_t, uint8_t>(..., CMPMODE::EQ, 2048)`, converts the compare mask to a 0/1 float flag vector with `Select`, then uses `ReduceMax(..., calIndex=true)` to return the first matching table index. Only the final index/state extraction stays scalar. For multi-request lookup, work is flattened as `req_id * ceil(Q / 64) + query_tile_id`. A core reloads the UB table only when its assigned work moves to another request.

The update kernel copies each request's `table_keys/table_states` to UB, applies random updates to UB, and writes the entire 8KB `table_states[r]` back to HBM by `DataCopy`, avoiding `GlobalTensor::SetValue` DCache/cacheline visibility issues for the resident state table.

## Optimization notes for Ascend 910B modeling

- Current lookup is scan-based: each query key performs up to 2048 key comparisons. This is easy to validate, but its cost scales as `O(R * Q * 2048)`.
- The lookup path is still scan-based, but the inner mask parsing and candidate verification have been moved from scalar loops to `Select`/`ReduceMax`. The remaining scalar work is query load, hit/miss branch, state load, and output staging.
- If the real KVCache key space can be made dense or bucketed, a direct index or open-addressed hash table in HBM should be modeled next. That changes lookup from scanning 2K entries to a small fixed number of HBM loads.
- For scan mode, keeping one request's 2K table in UB is a good fit: keys plus states are 16KB. Avoid caching multiple requests in UB unless the table format is compressed.
- Update is intentionally per-request sequential. Parallelizing updates within the same request would need conflict handling for duplicate query keys and deterministic last-writer semantics.
