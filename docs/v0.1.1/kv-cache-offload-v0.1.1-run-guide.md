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
| 真实 HBM index 算子 | `torch.ops._C_ascend.asu_hbm_index_lookup`、`asu_hbm_index_maintain_aicpu` 必须已注册 |
| 模型 | 走 SFA / DeepSeek sparse attention 的模型（V3.2 类），使用 `npu_sparse_flash_attention` |
| 分支 | vllm-ascend `feat/kv-offload-v011-compact-sfa`（含 offload block carve-out 改动） |

## 3. 环境变量

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1      # 开 v0.1.1 compact 路径
export VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1  # 必须 > 0，否则启动报错
export MICROKV_SOCKET=/tmp/microkv.sock             # 与 MicroKV server 监听地址一致
```

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
# 终端 1：启动 MicroKV server
<microkv-server> --socket /tmp/microkv.sock

# 终端 2：启动 vLLM
VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1 \
VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=1 \
MICROKV_SOCKET=/tmp/microkv.sock \
python -m vllm.entrypoints.openai.api_server \
  --model <deepseek-v3.2-sfa-model> \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1
```

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
  tests.ut.attention.test_offload_kv_cache_v0_ownership -v
```

覆盖：carve-out 算术、offload 尾部与 scheduler 范围不相交、registry 划分、启动 fail-fast。

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
| lookup/maintain op 不存在 | `_C_ascend` 算子未注册 |
| MicroKV miss after compact lookup | prefill 未写入 / socket 不通 / key 不匹配 |
| compact SFA topk token outside supported prefill range | topk 选到 `>= prefill_len` 的已生成 token（见 §7 限制 2） |
| does not support DSA CP / Sparse C8 indexer | 关闭对应特性 |
