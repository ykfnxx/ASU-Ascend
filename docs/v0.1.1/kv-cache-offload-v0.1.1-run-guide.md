# v0.1.1 KV Cache Offload Compact SFA 运行与测试指南

> 状态：Guide
> 配套设计：[kv-cache-offload-v0.1.1-block-table-adapt-design.md](./kv-cache-offload-v0.1.1-block-table-adapt-design.md)
> 适用路径：vllm-ascend SFA eager `DecodeOnly` 调试路径

本文说明在真实 Ascend NPU 环境上启用并验证 v0.1.1 compact SFA offload 路径所需的参数、前置条件和排障方法。相关框架改动位于 `vllm-ascend` 仓库分支 `feat/kv-offload-v011-compact-sfa`。

## 1. 适用范围

compact SFA 路径当前**只覆盖**：

- eager 模式（无图捕获）
- `DecodeOnly` attn state
- 单请求（`req_num = 1`）的 lookup / maintain
- topk 只落在 prefill backing store 内的 token（见 §7 限制）

**不支持**（命中即报错）：混合 prefill/decode batch、spec decode 多 query、DSA CP、Sparse C8 indexer、图捕获、非 DecodeOnly。

## 2. 前置条件

| 依赖 | 说明 |
|---|---|
| MicroKV server | 需先启动并监听 `MICROKV_SOCKET`；prefill 写入、decode miss 读取都走它 |
| HBM index 算子 | 默认用真实算子 `torch.ops._C_ascend.asu_hbm_index_lookup` / `asu_hbm_index_maintain_aicpu`（须已注册）。bring-up 阶段可用 `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` 切换为纯 Python 参考实现，此时**不要求**真实新算子（见 §3） |
| 模型 | 走 SFA / DeepSeek sparse attention 的模型（V3.2 类），使用 `npu_sparse_flash_attention` |
| 分支 | vllm-ascend `feat/kv-offload-v011-compact-sfa`（含 offload block carve-out 改动） |

## 3. 环境变量

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1      # 开 v0.1.1 compact 路径
export VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1  # 必须 > 0，否则启动报错
export MICROKV_SOCKET=/tmp/microkv.sock             # 与 MicroKV server 监听地址一致

# 可选：bring-up 阶段用纯 Python 参考算子，绕开尚未注册的真实新算子
export VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1
```

`VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` 时，lookup/maintain 用 `offload_kv_cache_v0_ref_ops.py` 的纯 Python 实现（语义对齐 kernel），此时无需 `_C_ascend.asu_hbm_index_*`；但 SFA / lightning indexer 仍是既有真实算子，仍需 NPU。仅用于验证框架接线，不追求性能。

注意：

- **不要**同时开 `VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE=1`。那是 v0.1 旁路校验（独立 tensor + 一致性比较），与 compact 是两条不同路径。
- `VLLM_ASCEND_KV_OFFLOAD_V0_CAPACITY` **当前未接入 compact 路径**，设了不生效——manager 构造未传 `slot_count`，`SLOT_COUNT` 恒为默认 `10240`。要改容量需改框架构造代码。

## 4. vLLM 启动参数

| 参数 | 必需性 | 原因 |
|---|---|---|
| `--enforce-eager` | 必需 | compact/validate 路径要求 eager；图捕获会 raise |
| `--gpu-memory-utilization` | 建议显式给足 | carve-out 会扣掉 offload pool，需保证普通 block 仍够用 |
| `--max-num-seqs 1` | 建议 | 当前 lookup/maintain 仅覆盖单请求；建议与 `MAX_PINNED_REQS` 一致 |

**不能**开启：aclgraph / cudagraph（非 NONE 且非 enforce_eager 会 raise）、DSA CP、Sparse C8 indexer。

## 5. 容量与 block 预算

carve-out 后的关系：

```text
compact_blocks_per_req = ceil(SLOT_COUNT / block_size)          # 默认 ceil(10240/128) = 80
R (offload_reserved_blocks) = MAX_PINNED_REQS * compact_blocks_per_req
normal_allocator_blocks = total_vllm_blocks - R
```

约束：

1. `normal_allocator_blocks > 0`，否则 `determine_available_memory` fail-fast 抛错（预留 offload 内存后普通 K/V 无空间）。
2. 初始化时断言 `tensor_blocks == kv_cache_config.num_blocks + R`；触发说明 page/split 对齐异常，需排查，不是正常情况。
3. 显存占用不变：预留出的 `R` 块内存以 tensor 尾部的形式还回，总 HBM footprint 与不开 offload 时一致。

隔离保证（本次改动后）：offload pool 是物理 tensor 尾部 `[num_blocks, num_blocks + R)`，scheduler 只分配 `[0, num_blocks)`，两者构造上不相交，普通 `slot_mapping` / `block_table` 物理上不可能落到 offload block。

## 6. 启动示例

```bash
# 终端 1：启动 MicroKV server（先 make 构建，产出 build/kv_stored）
cd MicroKV && make
./build/kv_stored --socket /tmp/microkv.sock

# 终端 2：启动 vLLM
export PYTHONPATH=/path/to/MicroKV/python:$PYTHONPATH   # 让 microkv 客户端可导入
export MICROKV_SOCKET=/tmp/microkv.sock                 # 须与服务端 --socket 一致
export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1
export VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1

vllm serve <deepseek-v3.2-sfa-model> \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1
```

> `vllm serve <model>` 与 `python -m vllm.entrypoints.openai.api_server --model <model>` 等价（两者最终都调用同一个 `run_server`）；若 `vllm` 不在 PATH，可退回后者。

## 7. 已知限制（属于设计范围，非缺陷）

1. **CAPACITY 未接线**：见 §3。
2. **topk 越界/padding 直接报错**：`_collect_compact_query_tokens` 对 `token_pos < 0` 或 `>= prefill_len` 会 raise。因此稳定可跑的是"topk 只落在 prefill token"的场景。多步 decode 一旦 topk 选到已生成 token（`>= prefill_len`）就会报错——建议先用**单步 / 短 decode** 验证坐标转换与隔离。
3. **单请求**：`req_num > 1` 的 batched lookup/maintain 是后续项。
4. **compact `actual_seq_lengths_kv` 固定为 `compact_blocks_per_req * block_size`**（默认 10240），因果掩码依赖改写前的原始 topk 保证。

## 8. 验证清单

本机（无 NPU）已覆盖的纯 Python 单测：

```bash
cd vllm-ascend
python3 -m unittest \
  tests.ut.attention.test_offload_kv_cache_v0_carveout \
  tests.ut.attention.test_offload_kv_cache_v0_ownership \
  tests.ut.attention.test_offload_kv_cache_v0_ref_ops -v
```

覆盖：carve-out 算术、offload 尾部与 scheduler 范围不相交、registry 划分、启动 fail-fast；以及参考 lookup/maintain 的命中/miss/重复 token/回补/protected 语义与 index↔slot_to_index 双向一致性。

需在 Ascend 环境补充的 e2e：

1. compact SFA 输出与原始 SFA 在同一 topk 下对齐。
2. SFA kernel 实际读取 offload 尾部 physical blocks（而非普通 request blocks）。
3. 普通请求 block table / slot_mapping 全程不含 offload block（现在应物理不可能，可加断言监控）。
4. request 结束后 offload pinned blocks 回到 `OffloadBlockPool` 并可复用。

## 9. 排障速查

| 报错 / 现象 | 可能原因 |
|---|---|
| `VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA requires MAX_PINNED_REQS > 0` | 未设 `MAX_PINNED_REQS` 或设为 0 |
| `KV offload v0 requires eager mode` | 未加 `--enforce-eager` / 开了 cudagraph |
| offload pool leaves no memory for normal K/V blocks | `R >= total_blocks`，KV 显存预算太小或 `MAX_PINNED_REQS` 太大 |
| `offload tensor blocks ... != scheduler blocks ... + reserved` | page/split 对齐异常，需排查 tensor 分配 |
| lookup/maintain op 不存在 | `_C_ascend` 算子未注册；bring-up 阶段可设 `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` 用参考实现 |
| `reference maintain cannot reclaim enough free slots` | 用参考算子时可回收的非 protected 已占用 slot 少于 `free_head`，属异常状态需排查 |
| MicroKV miss after compact lookup | prefill 未写入 / socket 不通 / key 不匹配 |
| compact SFA topk token outside supported prefill range | topk 选到 `>= prefill_len` 的已生成 token（见 §7 限制 2） |
| does not support DSA CP / Sparse C8 indexer | 关闭对应特性 |

## 10. 改动文件清单

范围：vllm-ascend 仓库上游基线 `36e15a2fdc` 之后的全部 offload 提交，分支 `feat/kv-offload-v011-compact-sfa`。

涉及提交（旧→新）：

| 提交 | 说明 |
|---|---|
| `6e867c102e` | Add v0 KV offload validation path（v0 旁路校验） |
| `7a139ee59a` | wire asu hbm index real ops（接入真实 lookup/maintain 算子） |
| `a374ca2a95` | add kv offload compact sfa path（v0.1.1 compact SFA） |
| `38beaf1d40` | carve offload pinned blocks out of the normal KV allocator（block carve-out） |
| `db009081c2` | add pure-Python reference HBM index ops（纯 Python 参考算子） |

### 10.1 算子内核 `csrc/`

| 文件 | 类型 | 责任 |
|---|---|---|
| `csrc/asu_hbm_index_lookup/`（`*_torch_adpt.h`、`op_host/*`、`op_kernel/asu_hbm_index_lookup.cpp`，共 7 个文件） | 新增 | HBM index lookup 算子：host tiling/proto/def + AICore kernel + torch 适配 |
| `csrc/asu_hbm_index_maintain_aicpu/`（`README.md`、`*_torch_adpt.h`、`op_host/*`、`op_kernel/*`，共 7 个文件） | 新增 | HBM index maintain（AICPU）算子及 torch 适配 |
| `csrc/torch_binding.cpp` | 修改 | 注册上述两个算子到 `torch.ops._C_ascend` |

### 10.2 框架 `vllm_ascend/`

| 文件 | 类型 | 责任 |
|---|---|---|
| `vllm_ascend/attention/offload_kv_cache_v0.py` | 新增 | offload manager：MicroKV 持久化、旁路校验、compact 输入准备、resident 窗口、算子调用 |
| `vllm_ascend/attention/offload_kv_cache_v0_ownership.py` | 新增 | block owner registry、静态 carve-out、compact block table、reserved 块/字节纯函数 |
| `vllm_ascend/attention/offload_kv_cache_v0_ref_ops.py` | 新增 | 纯 Python 参考 lookup/maintain 及 torch 包装层 |
| `vllm_ascend/attention/sfa_v1.py` | 修改 | indexer 后、SFA 前接入 offload hook：persist / validate / compact 输入切换 |
| `vllm_ascend/attention/utils.py` | 修改 | offload 路径所需的元数据/工具改动 |
| `vllm_ascend/ascend_forward_context.py` | 修改 | 通过 `_EXTRA_CTX` 透传 offload manager 与 capturing 标志 |
| `vllm_ascend/envs.py` | 修改 | 开关：`VALIDATE` / `COMPACT_SFA` / `MAX_PINNED_REQS` / `MICROKV_SOCKET` / `CAPACITY` / `REF_HBM_OPS` |
| `vllm_ascend/worker/model_runner_v1.py` | 修改 | 构造 manager、注册 offload pool、block carve-out、参考算子注入、release |
| `vllm_ascend/worker/worker.py` | 修改 | `determine_available_memory` 预留 offload pool 内存 + 启动 fail-fast |

### 10.3 单元测试 `tests/`

| 文件 | 类型 | 责任 |
|---|---|---|
| `tests/ut/attention/test_offload_kv_cache_v0.py` | 新增 | manager 核心：MicroKV record、compact 输入、physical slot 写入 |
| `tests/ut/attention/test_offload_kv_cache_v0_ownership.py` | 新增 | block ownership、compact block table、slot 映射 |
| `tests/ut/attention/test_offload_kv_cache_v0_carveout.py` | 新增 | carve-out 算术、尾部与 scheduler 范围不相交、启动 fail-fast |
| `tests/ut/attention/test_offload_kv_cache_v0_ref_ops.py` | 新增 | 参考 lookup/maintain 语义与双向一致性 |
| `tests/ut/attention/test_sfa_v1_compact_static.py` | 新增 | compact wiring 静态检查（indexer/SFA 输入切换、envs/构造） |
| `tests/ut/ops/test_asu_hbm_index_csrc_wiring.py` | 新增 | csrc 算子注册/绑定检查 |
| `tests/ut/attention/test_sfa_v1.py` | 修改 | 补充 offload hook 相关用例 |
| `tests/ut/attention/test_attention_v1.py` | 修改 | 相关回归调整 |

### 10.4 文档（ASU-Ascend 仓库）

| 文件 | 类型 | 责任 |
|---|---|---|
| `docs/v0.1.1/kv-cache-offload-v0.1.1-run-guide.md` | 新增 | 本运行与测试指南 |

汇总：vllm-ascend 侧共 32 个文件（新增 23、修改 9；其中 csrc 两个算子目录各 7 个新增文件）；ASU-Ascend 侧 1 个新增文档。
