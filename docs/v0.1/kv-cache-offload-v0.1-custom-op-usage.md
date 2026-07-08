# v0.1 HBM Index Custom Op 编译与使用方法

> 状态：Guide
> 配套设计：[kv-cache-offload-v0.1-real-ops-design.md](./kv-cache-offload-v0.1-real-ops-design.md)
> 适用路径：vllm-ascend SFA eager 调试路径；默认使用真实 `_C_ascend` 自定义算子

本文记录 v0.1 / v0.1.1 当前阶段如何沿 vllm-ascend 自定义算子路线编译、安装和调用新增的 HBM index 算子。这里讨论的是正式 custom-op / aclnn / OPP 路线，不是 `simu/hbm_lookup_update` 那种独立 pybind eager direct-launch 调试路线。

## 1. 目标算子

| 算子 | 公开 Python 名称 | 实现目标 | 说明 |
|---|---|---|---|
| lookup | `torch.ops._C_ascend.asu_hbm_index_lookup` | AICore custom op | 根据 `query_index` 查询 / 分配 HBM resident slot，返回 `slot_out` |
| maintain | `torch.ops._C_ascend.asu_hbm_index_maintain_aicpu` | AICPU custom op | 根据 `last_query_slots` 保护集合回收 slot，把 `free_head` 恢复到可继续 lookup 的状态 |

lookup 已确认走真实 AICore 算子；maintain 必须走 AICPU 版本，不能误绑到 AICore `asu_hbm_index_maintain` 参考实现。

## 2. 接入层次

需要同时满足两层接入：

1. CANN custom OPP / aclnn 层：`csrc/build.sh` 编译 operator host/kernel，生成并安装 `CANN-custom_ops*.run` 到 `vllm_ascend/_cann_ops_custom`。
2. PyTorch binding 层：`vllm_ascend.vllm_ascend_C` 导入后，在 `_C_ascend` namespace 注册 `asu_hbm_index_lookup` 和 `asu_hbm_index_maintain_aicpu`。

只完成第一层时，CANN 能找到算子二进制，但 Python 侧没有 `torch.ops._C_ascend.*` 调用入口。只完成第二层时，Python schema 存在，但 `EXEC_NPU_CMD(aclnn...)` 运行时可能找不到 aclnn/custom OPP 实现。

## 3. vllm-ascend 当前代码位置

目标分支：`feat/kv-offload-v011-compact-sfa`。

| 文件 / 目录 | 作用 |
|---|---|
| `csrc/asu_hbm_index_lookup/` | lookup 的 `op_host/`、`op_kernel/`、torch adapter |
| `csrc/asu_hbm_index_maintain_aicpu/` | AICPU maintain 的 `op_host/`、`op_kernel/`、torch adapter |
| `csrc/torch_binding.cpp` | 注册 `_C_ascend.asu_hbm_index_lookup` / `_C_ascend.asu_hbm_index_maintain_aicpu` |
| `csrc/build_aclnn.sh` | `pip install` 时选择要打包的 custom ops |
| `setup.py` | `COMPILE_CUSTOM_KERNELS=1` 时先运行 `build_aclnn.sh`，再编译 `vllm_ascend_C` |
| `vllm_ascend/platform.py` | 设置 `ASCEND_CUSTOM_OPP_PATH` 到 `vllm_ascend/_cann_ops_custom/vendors/vllm-ascend` |
| `vllm_ascend/utils.py` | `enable_custom_op()` 导入 `vllm_ascend_C` 并注册 torch ops |
| `vllm_ascend/attention/offload_kv_cache_v0.py` | `_call_lookup()` / `_call_maintain()` 的实际调用点 |

当前需要重点确认：`build_aclnn.sh` 的 A2 / A3 custom op 列表必须包含这两个新算子，否则正常 `pip install` 不会把它们打进 OPP 包。

## 4. 正常编译安装流程

在有 Ascend NPU / CANN / torch-npu 的机器上执行：

```bash
cd /path/to/vllm-ascend

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
# 如果环境使用 set_env.sh，则改为：
# source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SOC_VERSION=ascend910b1        # A2 示例；按 npu-smi 实际型号设置
export COMPILE_CUSTOM_KERNELS=1

python3 -m pip install --no-build-isolation -e .
```

期望产物：

```text
vllm_ascend/vllm_ascend_C*.so
vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/...
```

`setup.py` 会在 `COMPILE_CUSTOM_KERNELS=1` 时执行两件事：

1. 调 `csrc/build_aclnn.sh` 编译并安装 custom OPP。
2. 编译 `vllm_ascend_C`，使 `torch.ops._C_ascend.*` schema / impl 可注册。

## 5. 单独编译两个算子的临时流程

如果还没把两个新算子加入 `build_aclnn.sh` 的默认列表，可以先在 Ascend 环境中单独验证 OPP 编译：

```bash
cd /path/to/vllm-ascend/csrc

bash build.sh \
  -n "asu_hbm_index_lookup;asu_hbm_index_maintain_aicpu" \
  -c ascend910b

./output/CANN-custom_ops*.run \
  --install-path=/path/to/vllm-ascend/vllm_ascend/_cann_ops_custom
```

注意：这只覆盖 CANN custom OPP 层。Python 侧仍需要安装 / 编译 vllm-ascend，使 `vllm_ascend_C` 包含 `csrc/torch_binding.cpp` 中的新注册。

## 6. 运行时注册检查

在 Ascend 运行环境中检查：

```python
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import enable_custom_op
import torch

NPUPlatform.import_kernels()
assert enable_custom_op()

print(hasattr(torch.ops._C_ascend, "asu_hbm_index_lookup"))
print(hasattr(torch.ops._C_ascend, "asu_hbm_index_maintain_aicpu"))
print(torch.ops._C_ascend.asu_hbm_index_lookup)
print(torch.ops._C_ascend.asu_hbm_index_maintain_aicpu)
```

两个 `hasattr` 都应为 `True`。如果 `enable_custom_op()` 返回 `False`，先排查 `vllm_ascend_C` 是否成功编译 / 安装，而不是先看 OPP 包。

也可以直接检查 OPP 安装路径：

```bash
find vllm_ascend/_cann_ops_custom/vendors/vllm-ascend -iname '*asu*hbm*' -print
find vllm_ascend/_cann_ops_custom/vendors/vllm-ascend -iname '*maintain*' -print
```

## 7. 算子级 smoke 调用

最小输入形状与当前 kernel 常量保持一致：

```text
index          : [1, 128 * 1024], int32, NPU
slot_to_index  : [1, 10 * 1024], int32, NPU
free_slots     : [1, 2 * 1024], int32, NPU
free_head      : [1], int32, NPU
query_index    : [1, 2 * 1024], int32, NPU
last_query_slots: [1, 2 * 1024], int32, NPU
```

示例：

```python
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
print(slot_out.shape, int(free_head[0].item()))

last_query_slots = slot_out
torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
    index,
    slot_to_index,
    free_slots,
    free_head,
    last_query_slots,
    1,
    0,
)
torch.npu.synchronize()
print(int(free_head[0].item()))
```

期望现象：

1. lookup 返回 `[1, 2048]` 的 `slot_out`。
2. 首次全 miss 时 `free_head` 从 `0` 增加到 `2048`。
3. maintain 同步执行后，若存在足够非 protected resident slot，`free_head` 回到 `0`。

## 8. vLLM 路径中的调用方式

框架侧调用点已经收敛在两个方法里：

```python
slot_out = torch.ops._C_ascend.asu_hbm_index_lookup(
    state.index,
    state.slot_to_index,
    state.free_slots,
    state.free_head,
    query_index,
    1,
)

torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
    state.index,
    state.slot_to_index,
    state.free_slots,
    state.free_head,
    state.last_query_slots,
    1,
    maintain_seed,
)
```

v0.1.1 compact 路径的顺序应保持为：

1. lightning indexer 产生 `topk_indices`。
2. Python manager 去重 / padding，构造 `[1, 2048]` 的 `query_index`。
3. 同步调用 lookup，得到 `slot_out`，lookup 内部会为 miss token 分配 slot 并推进 `free_head`。
4. 根据 `slot_out` 把 MicroKV 中的 K/V 加载到 offload HBM slot。
5. `state.last_query_slots.copy_(slot_out)`。
6. 同步调用 maintain。当前实现可在 `free_head > 0` 时调用；如果为了调试调用路径，也可以每次 lookup 后都调用，AICPU maintain 在 `free_head == 0` 时应快速返回。
7. 用 compact block table / remapped topk 调用 SFA。

调试时建议打开 trace：

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1
```

日志应能看到 lookup 前后 `free_head` / free slot 数量变化，以及 maintain 回收数量。

## 9. vLLM 启动示例

使用真实新算子时不要打开参考实现开关：

```bash
export MICROKV_SOCKET=/tmp/microkv.sock
export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1
export VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1
export VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1
unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS

vllm serve <deepseek-sfa-model> \
  --enforce-eager \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.9
```

如果只想先验证框架接线、暂不验证真实新算子，则打开：

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1
```

此时 lookup / maintain 走纯 Python 参考实现，不依赖 `_C_ascend.asu_hbm_index_*`，但 SFA / lightning indexer 仍依赖 NPU 真实算子。

## 10. 当前必须补齐 / 验证的点

1. `csrc/build_aclnn.sh` 的 A2 / A3 custom op 列表需要加入：

```text
asu_hbm_index_lookup
asu_hbm_index_maintain_aicpu
```

否则 `COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e .` 不会自动把这两个算子打进 OPP 包。

2. AICPU maintain 的 packaging 需要在真实 Ascend/CANN 环境确认。

当前目录里有：

```text
csrc/asu_hbm_index_maintain_aicpu/op_kernel/asu_hbm_index_maintain_aicpu.cpp
csrc/asu_hbm_index_maintain_aicpu/op_kernel/asu_hbm_index_maintain_aicpu_kernel.aicpu
```

vllm-ascend 现有通用 custom-op build 主要面向 AICore `op_kernel/*.cpp`。如果 build 后 OPP 包里没有 maintain AICPU 二进制，不能靠 Python 层修复，需要按 CANN AICPU custom op 的标准编译 / 打包方式调整目录或 CMake。

3. 本机没有 NPU，无法完成 e2e 结论。

本地能做的只是静态检查和 Python 语法检查；真正的 build、`enable_custom_op()`、算子 smoke、vLLM compact SFA 运行都必须在 Ascend/CANN 机器上完成。

## 11. 排障速查

| 现象 | 优先排查 |
|---|---|
| `hasattr(torch.ops._C_ascend, "asu_hbm_index_lookup") == False` | `vllm_ascend_C` 未编译 / 未导入；`enable_custom_op()` 返回 False；`csrc/torch_binding.cpp` 注册未进入当前安装包 |
| `aclnnAsuHbmIndexLookup` 相关符号或运行时找不到 | custom OPP 未编译 / 未安装；`ASCEND_CUSTOM_OPP_PATH` 未指到 `vllm_ascend/_cann_ops_custom/vendors/vllm-ascend` |
| lookup 存在但 maintain 不存在 | `csrc/torch_binding.cpp` 或 maintain torch adapter 未进入编译；确认 `_aicpu` 后缀没有写错 |
| maintain schema 存在但运行失败 | AICPU kernel 未被正确打包；检查 OPP 包中是否有 maintain AICPU 产物 |
| `enable_custom_op()` 返回 False | 确认不是 Ascend 950/A5 禁用路径；确认 `COMPILE_CUSTOM_KERNELS=1` 且扩展编译成功 |
| vLLM 启动时报 eager 相关错误 | 必须加 `--enforce-eager`，当前 offload 路径不支持 graph |
| 没有 lookup / maintain trace 日志 | 确认 `VLLM_ASCEND_KV_OFFLOAD_V0_TRACE_INDEX_OPS=1`，且请求实际进入 decode/SFA/offload 路径 |

## 12. 官方资料

- CANN simple custom operator project：说明 aclnn custom op 项目会生成 op host/kernel、`libcust_opapi.so` 和可安装的 custom operator package。
  <https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0101.html>
- CANN PyTorch custom operator adaptation：说明 PyTorch 场景可通过 torch.library / pybind 做 kernel launch，也可通过 single-operator API 或 graph mode 接入。
  <https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0057.html>
- `ASCEND_CUSTOM_OPP_PATH`：说明 custom operator package / dynamic library 的安装路径搜索规则。
  <https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0147.html>
