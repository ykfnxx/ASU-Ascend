# v0.1 HBM Index 算子编译与框架调用方法

> 状态：Guide
> 配套设计：[kv-cache-offload-v0.1-real-ops-design.md](./kv-cache-offload-v0.1-real-ops-design.md)
> 适用路径：vllm-ascend SFA eager direct 调试路径

当前 v0.1 bring-up 不再依赖 vllm-ascend custom OPP / opdef 路线调用新增 HBM index 算子。lookup 和 maintain 都降级为 ASU direct `.so`，再按 vllm 当前 `lookup_op` / `maintain_op` hook 的调用格式注入框架。

| 算子 | 编译位置 | 框架内调用方式 | Python callable 格式 |
|---|---|---|---|
| lookup | `ASU-Ascend/ops` 的 `lookup_aiv` | `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB` 注入 `lookup_op` | `lookup_op(index, slot_to_index, free_slots, free_head, query_index, req_num) -> slot_out` |
| maintain | `ASU-Ascend/ops` 的 `maintain_aicpu` | `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB` 注入 `maintain_op` | `maintain_op(index, slot_to_index, free_slots, free_head, last_query_slots, req_num, seed) -> None` |

`vllm-ascend/csrc/asu_hbm_index_lookup/` 和 `vllm-ascend/csrc/asu_hbm_index_maintain_aicpu/` 里的 custom-op 目录仍可保留，但当前可运行路径不要求 lookup OPP、maintain opdef 或 `_C_ascend.asu_hbm_index_*` 注册。

## 1. 环境准备

在 Ascend/CANN 机器上执行。A2 示例：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

如果机器使用 `set_env.sh`，改为：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

ASU direct build 默认使用 `Ascend910B3` 和 `NPU_ARCH=dav-2201`。如果目标机器不同，按机器实际值设置：

```bash
export NPU_ARCH=<target-aicpu-arch>
```

## 2. 编译 lookup direct `.so`

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops

bash build.sh lookup_aiv Ascend910B3

find build/lookup_aiv -name 'libasu_hbm_index_lookup_aiv*.so' -print
```

常见产物路径为：

```text
/home/solidyang/workspace/ASU-Ascend/ops/build/lookup_aiv/lib/libasu_hbm_index_lookup_aiv.so
```

导出的 C ABI：

```c
void asu_hbm_index_lookup_do(
    uint32_t blockDim,
    void* stream,
    void* index,
    void* slotToIndex,
    void* freeSlots,
    void* freeHead,
    void* queryIndex,
    void* slotOut,
    uint32_t reqNum);
```

vllm-ascend 的 direct wrapper 内部会申请 `slot_out = torch.empty_like(query_index)`，调用该 ABI，然后返回 `slot_out`，所以框架层仍保持当前 `_call_lookup()` 的返回值语义。

## 3. 编译 maintain direct `.so`

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops

bash build.sh maintain_aicpu Ascend910B3

find build/maintain_aicpu -name 'libasu_hbm_index_maintain_aicpu*.so' -print
```

常见产物路径为：

```text
/home/solidyang/workspace/ASU-Ascend/ops/build/maintain_aicpu/lib/libasu_hbm_index_maintain_aicpu.so
```

导出的 C ABI：

```c
void asu_hbm_index_maintain_do(
    uint32_t blockDim,
    void* stream,
    void* index,
    void* slotToIndex,
    void* freeSlots,
    void* freeHead,
    void* lastQuerySlots,
    uint32_t reqNum,
    uint32_t seed);
```

## 4. 在框架内启用 direct 调用

启动 vLLM 前设置：

```bash
export MICROKV_SOCKET=/tmp/microkv.sock
export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1
export VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1
export VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1

unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS
export VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB=\
/home/solidyang/workspace/ASU-Ascend/ops/build/lookup_aiv/lib/libasu_hbm_index_lookup_aiv.so
export VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB=\
/home/solidyang/workspace/ASU-Ascend/ops/build/maintain_aicpu/lib/libasu_hbm_index_maintain_aicpu.so
```

然后用 eager 模式启动：

```bash
vllm serve <deepseek-sfa-model> \
  --enforce-eager \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.9
```

框架内调用关系：

```text
NPUModelRunner
  -> OffloadKVCacheV0Manager(
       lookup_op=load_direct_lookup_op($VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB),
       maintain_op=load_direct_maintain_op($VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB)
     )

OffloadKVCacheV0Manager._call_lookup()
  -> lookup_op(state.index, state.slot_to_index, state.free_slots,
               state.free_head, query_index, 1)

OffloadKVCacheV0Manager._call_maintain()
  -> maintain_op(state.index, state.slot_to_index, state.free_slots,
                 state.free_head, state.last_query_slots, 1, maintain_seed)
```

`VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` 优先级更高；如果打开它，lookup / maintain 都会走纯 Python 参考实现，不会使用 direct `.so`。

## 5. direct loader 检查

lookup loader：

```python
import importlib.util
import os
from pathlib import Path

module_path = Path("/home/solidyang/workspace/vllm-ascend/csrc/asu_hbm_index_lookup/tmp/direct_lookup.py")
spec = importlib.util.spec_from_file_location("asu_hbm_index_direct_lookup", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

lookup_op = module.load_direct_lookup_op(
    os.environ["VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB"]
)
print(lookup_op)
```

maintain loader：

```python
import importlib.util
import os
from pathlib import Path

module_path = Path("/home/solidyang/workspace/vllm-ascend/csrc/asu_hbm_index_maintain_aicpu/tmp/direct_maintain.py")
spec = importlib.util.spec_from_file_location("asu_hbm_index_direct_maintain", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

maintain_op = module.load_direct_maintain_op(
    os.environ["VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB"]
)
print(maintain_op)
```

框架启动时应能看到：

```text
KV offload v0 using ASU direct lookup library ...
KV offload v0 using ASU direct AICPU maintain library ...
```

同时打开 trace 后，应能看到 lookup 前后和 maintain 前后的 `free_head` / free slot 数量变化：

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1
```

## 6. 最小算子级 smoke 调用

输入形状与当前 kernel 常量保持一致：

```text
index           [1, 128 * 1024], int32, NPU
slot_to_index   [1, 10 * 1024], int32, NPU
free_slots      [1, 2 * 1024], int32, NPU
free_head       [1], int32, NPU
query_index     [1, 2 * 1024], int32, NPU
last_query_slots[1, 2 * 1024], int32, NPU
```

示例：

```python
import importlib.util
import os
from pathlib import Path

import torch
import torch_npu

device = "npu:0"
INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
NOT_FOUND = -1

index = torch.full((1, INDEX_SIZE), NOT_FOUND, dtype=torch.int32, device=device)
slot_to_index = torch.full((1, SLOT_COUNT), NOT_FOUND, dtype=torch.int32, device=device)
free_slots = torch.arange(FREE_SLOT_COUNT, dtype=torch.int32, device=device).reshape(1, -1)
free_head = torch.zeros((1,), dtype=torch.int32, device=device)
query_index = torch.arange(QUERY_COUNT, dtype=torch.int32, device=device).reshape(1, -1)

lookup_path = Path("/home/solidyang/workspace/vllm-ascend/csrc/asu_hbm_index_lookup/tmp/direct_lookup.py")
lookup_spec = importlib.util.spec_from_file_location("asu_hbm_index_direct_lookup", lookup_path)
lookup_module = importlib.util.module_from_spec(lookup_spec)
lookup_spec.loader.exec_module(lookup_module)
lookup_op = lookup_module.load_direct_lookup_op(
    os.environ["VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB"]
)

maintain_path = Path("/home/solidyang/workspace/vllm-ascend/csrc/asu_hbm_index_maintain_aicpu/tmp/direct_maintain.py")
maintain_spec = importlib.util.spec_from_file_location("asu_hbm_index_direct_maintain", maintain_path)
maintain_module = importlib.util.module_from_spec(maintain_spec)
maintain_spec.loader.exec_module(maintain_module)
maintain_op = maintain_module.load_direct_maintain_op(
    os.environ["VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB"]
)

slot_out = lookup_op(index, slot_to_index, free_slots, free_head, query_index, 1)
torch.npu.synchronize()

maintain_op(index, slot_to_index, free_slots, free_head, slot_out, 1, 0)
torch.npu.synchronize()
```

## 7. 排障速查

| 现象 | 优先排查 |
|---|---|
| lookup 没走 direct `.so` | 确认 `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB` 已设置，且没有打开 `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` |
| maintain 没走 direct `.so` | 确认 `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB` 已设置，且没有打开 `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` |
| direct loader 加载失败 | 确认 env 指向真实存在的 ASU build `.so`，并且 CANN/torch-npu 环境已 source |
| `torch.npu.current_stream()` 相关失败 | direct wrapper 只能在 torch-npu / NPU runtime 环境中调用 |
| 没有 free slot trace 日志 | 确认 `VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1`，且请求实际进入 compact SFA / offload decode 路径 |
| vLLM 启动时报 graph/eager 错误 | 当前路径必须加 `--enforce-eager` |

## 8. 后续正式化方向

1. 当前 direct 路径优先满足 eager bring-up，不解决 graph capture。
2. 如果后续要回到 vllm-ascend packaged custom-op 路线，需要重新打通 lookup OPP 默认编译列表和 maintain AICPU opdef / packaging。
3. 在 Ascend/CANN 机器上验证：
   - ASU lookup / maintain `.so` 编译；
   - direct loader 可加载；
   - vLLM eager compact SFA 路径中 lookup 后同步调用 maintain；
   - trace 能观察 `free_head` 分配与回收。
