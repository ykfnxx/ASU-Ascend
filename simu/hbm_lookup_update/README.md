# hbm_lookup_update

Ascend 910B / CANN KernelLaunch + PyBind demo for a resident HBM key/state table.

## Semantics

The operator does **not** generate query keys internally. The caller passes `query_keys`.

Inputs:

- `table_keys`: `torch.int32`, NPU tensor, shape `[2048]`, resident HBM index keys. It can be unordered.
- `table_states`: `torch.int32`, NPU tensor, shape `[2048]`, resident HBM states. This tensor is updated in place.
- `query_keys`: `torch.int32`, NPU tensor, shape `[Q]`, external query keys.
- `new_states`: `torch.int32`, NPU tensor, shape `[Q]`, states used if a query position is selected for update.

Python call:

```python
states_out = hbm_lookup_update.lookup_random_update(
    table_keys, table_states, query_keys, new_states,
    seed=42, update_percent=5, block_dim=8, not_found=-1,
)
```

Meaning:

```cpp
// Kernel 1: multi-core lookup
for i in 0..Q-1:
    states_out[i] = table_states[j] if table_keys[j] == query_keys[i] else not_found

// Kernel 2: same stream, after lookup, single-core update
for pos in random_unique_positions(floor(Q * update_percent / 100), seed):
    key = query_keys[pos]
    if table_keys[j] == key:
        table_states[j] = new_states[pos]
```

`states_out` returns **pre-update** states.

Assumptions:

- `table_keys` should be unique. If duplicates exist, the first matching slot in table order is used.
- Key and state dtype is `int32`.
- Table size is fixed at 2048.

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
PYTHONPATH=$PWD/build:$PYTHONPATH QUERY_LEN=2048 BLOCK_DIM=8 ITERS=1000 python3 scripts/bench_lookup_update.py
```

## Design notes

The implementation uses two kernels under one Python function. This is deliberate: the lookup kernel is multi-core, while the update kernel is single-core and runs after lookup on the same stream. This avoids relying on cross-AI-Core synchronization inside one kernel and avoids multi-core random writes to the same resident HBM state table.

Lookup uses vector compare against the resident `table_keys`: each AI Core copies the 2K key/state table to UB, fills a 64-element query tile with the current query key, calls `Compare<int32_t, uint8_t>(..., CMPMODE::EQ, 64)`, then verifies candidate bytes and returns the corresponding state.

The update kernel copies `table_keys/table_states` to UB, applies random updates to UB, and writes the entire 8KB `table_states` back to HBM by `DataCopy`, avoiding `GlobalTensor::SetValue` DCache/cacheline visibility issues for the resident state table.
