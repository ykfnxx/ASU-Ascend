# v0.1 HBM Index 算子编译与框架调用方法

> 状态：Guide
> 配套设计：[kv-cache-offload-v0.1-real-ops-design.md](./kv-cache-offload-v0.1-real-ops-design.md)
> 适用路径：vllm-ascend SFA eager 调试路径

本文记录当前真实可操作的 v0.1 / v0.1.1 HBM index 算子接入方法。当前不要把两个算子视为同一条编译路径：

| 算子 | 当前编译方式 | 框架内调用方式 | 状态 |
|---|---|---|---|
| lookup | vllm-ascend `csrc` custom OPP + `vllm_ascend_C` torch binding | `torch.ops._C_ascend.asu_hbm_index_lookup(...)` | 当前框架真实 lookup 路径 |
| maintain | ASU-Ascend `ops` direct AICPU `.so` | 通过 `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB` 注入 `maintain_op` | 当前临时 eager 调试路径 |

`vllm-ascend/csrc/asu_hbm_index_maintain_aicpu/` 里仍保留了按 vllm custom-op 形态整理的 maintain 目录和 `_C_ascend.asu_hbm_index_maintain_aicpu` binding，但 AICPU opdef / packaging 还没有打通。现阶段不能把它当作可编译可运行的 maintain 主路径。

## 1. 环境准备

在 Ascend/CANN 机器上执行。A2 示例：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SOC_VERSION=ascend910b1
```

如果机器使用 `set_env.sh`，改为：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

`vllm-ascend/csrc/build.sh` 的 `-c` 参数使用 CANN custom-op build 的 compute-unit 名称：

| 机器 | `SOC_VERSION` 示例 | `csrc/build.sh -c` |
|---|---|---|
| A2 / 910B | `ascend910b1` / `ascend910b3` | `ascend910b` |
| A3 / 910C | `ascend910_9391` | `ascend910_93` |

## 2. 编译 lookup

lookup 需要两层产物：

1. CANN OPP 包：提供 `aclnnAsuHbmIndexLookup` 的 host/kernel 实现。
2. Python torch binding：提供 `torch.ops._C_ascend.asu_hbm_index_lookup` 调用入口。

当前 `vllm-ascend/csrc/build_aclnn.sh` 的默认列表还没有包含 `asu_hbm_index_lookup`，所以先用显式单算子命令编译 / 安装 lookup OPP。

```bash
cd /home/solidyang/workspace/vllm-ascend/csrc

bash build.sh -n asu_hbm_index_lookup -c ascend910b

./output/CANN-custom_ops*.run \
  --install-path=/home/solidyang/workspace/vllm-ascend/vllm_ascend/_cann_ops_custom
```

A3 环境把 `-c ascend910b` 改为 `-c ascend910_93`。

然后编译 vllm-ascend 的 Python 扩展，使 `csrc/torch_binding.cpp` 里的 `_C_ascend` schema / impl 注册进入 `vllm_ascend_C`：

```bash
cd /home/solidyang/workspace/vllm-ascend

export COMPILE_CUSTOM_KERNELS=1
python3 -m pip install --no-build-isolation -e .
```

注意：`COMPILE_CUSTOM_KERNELS=1` 会触发 `csrc/build_aclnn.sh` 编译默认 custom ops，并编译 `vllm_ascend_C`。在 `build_aclnn.sh` 正式加入 `asu_hbm_index_lookup` 前，如果安装流程覆盖了 `_cann_ops_custom`，需要在 `pip install -e .` 后重新执行一次上面的 lookup OPP 单独安装命令。

正式化后建议把 `asu_hbm_index_lookup` 加入 `csrc/build_aclnn.sh` 的 A2 / A3 `CUSTOM_OPS` 列表，这样 `COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e .` 就能一次性完成 lookup OPP + binding。

## 3. 编译 maintain

maintain 当前走 ASU direct AICPU `.so`，不走 vllm-ascend `csrc/build.sh -n asu_hbm_index_maintain_aicpu`。

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops

bash build.sh maintain_aicpu Ascend910B3

find build/maintain_aicpu -name 'libasu_hbm_index_maintain_aicpu*.so' -print
```

默认产物路径为：

```text
/home/solidyang/workspace/ASU-Ascend/ops/build/maintain_aicpu/lib/libasu_hbm_index_maintain_aicpu.so
```

如果目标机器的 AICPU arch 不是默认 `dav-2201`，编译前设置：

```bash
export NPU_ARCH=<target-aicpu-arch>
```

ASU direct AICPU `.so` 导出的 ABI 是：

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

vllm-ascend 当前通过 `ctypes` 绑定这个 ABI，并把它注入为 Python 层的 `maintain_op`。

## 4. 在框架内启用两个算子

启动 vLLM 前设置：

```bash
export MICROKV_SOCKET=/tmp/microkv.sock
export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1
export VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1
export VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1

unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS
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

框架内调用关系为：

```text
NPUModelRunner
  -> OffloadKVCacheV0Manager(
       lookup_op=None,
       maintain_op=load_direct_maintain_op($VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB)
     )

OffloadKVCacheV0Manager._call_lookup()
  -> torch.ops._C_ascend.asu_hbm_index_lookup(...)

OffloadKVCacheV0Manager._call_maintain()
  -> injected direct AICPU maintain_op(...)
```

也就是说：

- lookup 的框架可调用性依赖 `vllm_ascend_C` 已注册 `_C_ascend.asu_hbm_index_lookup`，并且 lookup OPP 已安装到 `vllm_ascend/_cann_ops_custom`。
- maintain 的框架可调用性依赖 `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB` 指向 ASU 编译出的 direct AICPU `.so`。
- `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` 优先级更高；如果打开它，lookup / maintain 都会走纯 Python 参考实现，不会使用真实 lookup 或 direct maintain。

## 5. 注册与可调用性检查

检查 lookup torch op：

```python
import torch
import torch_npu
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import enable_custom_op

NPUPlatform.import_kernels()
assert enable_custom_op()

assert hasattr(torch.ops._C_ascend, "asu_hbm_index_lookup")
print(torch.ops._C_ascend.asu_hbm_index_lookup)
```

检查 maintain direct loader：

```python
import os
from pathlib import Path
import importlib.util

module_path = Path("/home/solidyang/workspace/vllm-ascend/csrc/asu_hbm_index_maintain_aicpu/tmp/direct_maintain.py")
spec = importlib.util.spec_from_file_location("asu_hbm_index_direct_maintain", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

maintain_op = module.load_direct_maintain_op(
    os.environ["VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB"]
)
print(maintain_op)
```

框架启动时，如果 direct maintain env 生效，会打印类似日志：

```text
KV offload v0 using ASU direct AICPU maintain library ...; lookup still uses the real _C_ascend lookup op.
```

同时打开：

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1
```

应能看到 lookup 前后和 maintain 前后的 `free_head` / free slot 数量变化。

## 6. 最小算子级 smoke 调用

lookup 的最小输入形状与当前 kernel 常量保持一致：

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
import os
import torch
import torch_npu
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import enable_custom_op

NPUPlatform.import_kernels()
assert enable_custom_op()

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

slot_out = torch.ops._C_ascend.asu_hbm_index_lookup(
    index,
    slot_to_index,
    free_slots,
    free_head,
    query_index,
    1,
)
torch.npu.synchronize()

from pathlib import Path
import importlib.util

module_path = Path("/home/solidyang/workspace/vllm-ascend/csrc/asu_hbm_index_maintain_aicpu/tmp/direct_maintain.py")
spec = importlib.util.spec_from_file_location("asu_hbm_index_direct_maintain", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
maintain_op = module.load_direct_maintain_op(
    os.environ["VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB"]
)

maintain_op(index, slot_to_index, free_slots, free_head, slot_out, 1, 0)
torch.npu.synchronize()
```

## 7. 排障速查

| 现象 | 优先排查 |
|---|---|
| `hasattr(torch.ops._C_ascend, "asu_hbm_index_lookup") == False` | `COMPILE_CUSTOM_KERNELS=1` 是否生效；`vllm_ascend_C` 是否编译 / 导入；`enable_custom_op()` 是否返回 True |
| lookup torch op 存在但运行时报 `aclnnAsuHbmIndexLookup` 找不到 | lookup OPP 未安装，或 `ASCEND_CUSTOM_OPP_PATH` 未指向 `vllm_ascend/_cann_ops_custom/vendors/vllm-ascend` |
| maintain 编译 vllm custom op 报缺 opdef | 当前预期现象；现阶段 maintain 走 ASU direct AICPU `.so`，不走 vllm custom-op packaging |
| 启动后 maintain 没走 direct `.so` | 确认 `VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB` 已设置，且没有打开 `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` |
| 没有 free slot trace 日志 | 确认 `VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1`，且请求实际进入 compact SFA / offload decode 路径 |
| vLLM 启动时报 graph/eager 错误 | 当前路径必须加 `--enforce-eager` |

## 8. 后续正式化方向

1. 把 `asu_hbm_index_lookup` 加入 `vllm-ascend/csrc/build_aclnn.sh` 的 A2 / A3 默认列表。
2. 为 maintain 选择正式路径：
   - 要么继续沿 ASU direct AICPU `.so` 路线，把 direct loader 变成明确的调试/生产开关；
   - 要么补齐 CANN AICPU opdef / packaging，使 `torch.ops._C_ascend.asu_hbm_index_maintain_aicpu` 真正可编译、可安装、可运行。
3. 在 Ascend/CANN 机器上验证：
   - lookup OPP 编译 / 安装；
   - `enable_custom_op()` 后 lookup torch op 可见；
   - direct maintain `.so` 可加载；
   - vLLM eager compact SFA 路径中 lookup 后同步调用 maintain，并通过 trace 观察 `free_head` 回收。
