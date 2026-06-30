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

This repository may be developed on machines without an Ascend NPU. In that case, only static tests and script syntax checks can run locally; functional kernel validation requires the Ascend 950 CANN/PTA environment above.
