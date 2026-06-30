# ASU HBM Index Lookup SIMT PTA

This package is a standalone PTA implementation of ASU HBM index lookup for Ascend 950 SIMT. It does not modify the existing `ascend-ops/asu_hbm_index_lookup` or `ops/asu_hbm_index_lookup_aiv.cpp` sources.

## Contract

Inputs are contiguous `torch.int32` NPU tensors:

- `index`: `[req_num, 128K]`, maps index id to slot id, with `-1` for not found.
- `slot_to_index`: `[req_num, 10K]`, updated on allocation.
- `free_slots`: `[req_num, 2K]`, free slot ids.
- `free_head`: `[req_num]`, allocation head, updated in place.
- `query_index`: `[req_num, 2K]`, query ids.
- `req_num`: number of requests to process.

The function returns `slot_out` with the same shape as `query_index`. A miss is allocated from `free_slots[req_id, free_head[req_id]]`, and duplicate misses inside one query batch allocate only once. Allocation follows query order so the result matches `ops/scripts/asu_hbm_index_common.py::expected_lookup_allocate`.

Preconditions:

- query ids are in `[0, 128K)`;
- every request has enough free slots for its distinct misses;
- no other kernel mutates the same request rows concurrently.

## Build

Run on an Ascend 950 host with CANN, torch, torch_npu, and pybind11 installed:

```bash
cd pta-ops/asu_hbm_index_lookup_simt
SOC_VERSION=Ascend950 ./build.sh
```

`ASCEND_CANN_PACKAGE_PATH`, `ASCEND_HOME_PATH`, or `ASCEND_INSTALL_PATH` can point at the CANN toolkit root. The extension module is built under `build/`.

## Validate

On an Ascend 950 runtime host:

```bash
python3 scripts/validate_lookup_simt.py --build-dir build --device 0 --req-num 2 --pattern mixed
```

## Benchmark 10% Miss

To measure repeated whole-operator launch performance with 10% distinct misses and preloaded NPU inputs:

```bash
python3 scripts/bench_lookup_simt.py --build-dir build --device 0 --req-num 50 --rounds 100 --miss-ratio 0.10
```

The benchmark creates fresh states for every warmup and measured launch, uploads all input tensors to NPU before the timed loop, and keeps output tensors alive until synchronization. This preserves the 10% miss ratio for every measured launch without mixing free-slot reset or host-to-device copies into the measured section.

For `req_num=50`, one state is about 29 MiB. The default `--rounds 100 --warmup 10` preloads roughly 3.1 GiB of state. If the runtime host does not have enough free device memory, use batched preloading:

```bash
python3 scripts/bench_lookup_simt.py --build-dir build --device 0 --req-num 50 --rounds 100 --batch-rounds 20 --miss-ratio 0.10
```

The script reports both wall-clock average latency and NPU event average latency. Wall-clock time includes Python wrapper calls and host launch submission; NPU event time measures elapsed work on the NPU stream.

This repository may be developed on machines without an Ascend NPU. In that case, only static tests and script syntax checks can run locally; functional kernel validation requires the Ascend 950 CANN/PTA environment above.
