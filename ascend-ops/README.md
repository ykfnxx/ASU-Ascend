# ASU HBM Index vLLM-Ascend Operators

This directory re-implements the prototype operators from `ASU-Ascend/ops`
in the source layout used by vLLM-Ascend CANN custom operators.

The original prototype files are intentionally not modified. The two operators
here are independent source packages:

- `asu_hbm_index_lookup`: AICore lookup and miss allocation. It returns
  `slot_out` and updates `index`, `slot_to_index`, and `free_head` in place.
- `asu_hbm_index_maintain`: AICore maintenance and eviction. It follows the
  original AICPU eviction algorithm and updates the index state in place.

Each operator follows the vLLM-Ascend split:

- `op_host/*_def.cpp`: CANN operator definition.
- `op_host/*_proto.cpp`: shape and dtype inference.
- `op_host/*_tiling.{h,cpp}`: tiling data and block-dim selection.
- `op_kernel/*.cpp`: AscendC AICore kernel implementation.
- `*_torch_adpt.h`: adapter that can be included by a vLLM-Ascend binding.
- `torch_binding_asu_hbm_index.cpp`: a `TORCH_LIBRARY_FRAGMENT` that registers
  the two `_C_ascend` operators when added to a vLLM-Ascend pybind module build.

The current development environment cannot compile CANN/NPU code, so the
provided validation is static:

```bash
pytest -q ASU-Ascend/ascend-ops/tests/test_static_layout.py
```

To compile later, include this directory from a vLLM-Ascend `csrc` custom-op
build context or copy the two operator directories under `vllm-ascend/csrc`.
