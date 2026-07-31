# vllm-ascend KV Cache Offload 设计方案

> 状态：方案讨论稿，部分已决策
> 目标模型：使用 DSA/Sparse Flash Attention 的模型（DeepSeek-V3.2、GLM-5 等，当前先用 GLM-5.1 验证）
> 约束：尽量只修改 vllm-ascend；验证阶段不替换原 KV Cache，以外挂数据结构方式验证。

---

## 1. 背景与目标

### 1.1 目标

在 vllm-ascend 中适配一种 KV Cache offload 算法：

1. 每个请求的每个 attention layer 维护一个**新的 KV Cache 缓存池**（外挂数据结构）。
2. 在 `lightning indexer → SFA` 之间加入 **lookup 算子**，根据 top-K token id 查询该 token 在新缓存池中的 slot id。
3. SFA 根据 lookup 得到的 slot id 在新缓存池中寻址，完成 attention 计算。
4. 新缓存池的物理排布**立即与原 KV Cache 不同**：KV 不按请求/token 顺序排布，而是由新增 lookup 算子统一管理逻辑 token 到新缓存 slot 的映射。
5. 第一阶段只处理 **decode**，不处理 prefill。
6. 验证阶段不替换原有 KV Cache，仅作为外挂结构存在，保证原路径可回退。

### 1.2 当前硬件/环境约束

- 开发机没有 Ascend NPU，无法做完整 e2e 验证。
- 需要支持**无 NPU 的框架功能调试**，至少能验证数据流、shape、请求级隔离等逻辑。
- **关键约束：在 `lightning indexer → lookup → SFA` 这一计算流程中，CPU 完全 0 参与**。lookup 算子、`req_id_per_token` 构造、以及 `offload_slot_table` 访问都必须在 NPU 上完成；mock 调试路径与真实 NPU 路径必须严格隔离。
- **第一阶段不支持 CP（Context Parallel）路径**。开启 CP 时必须显式禁用 `offload_kv_cache` 或提前报错，避免进入未适配路径。
- **第一阶段不考虑量化和 scale**，包括 Sparse C8 的 indexer key scale 路径。

---

## 2. 当前 vllm-ascend SFA 路径梳理

### 2.1 KV Cache 物理布局

在 `NPUModelRunner._allocate/_reshape_kv_cache_tensors` 中，SFA 每层分配 3~4 个 tensor：

| tensor | 含义 | 形状 |
|---|---|---|
| `kv_cache[0]` | k_nope（ MLA 压缩后的 key/value 共用） | `(num_blocks, block_size, num_kv_heads, kv_lora_rank)` |
| `kv_cache[1]` | k_pe / rope | `(num_blocks, block_size, num_kv_heads, qk_rope_head_dim)` |
| `kv_cache[2]` | lightning indexer key | `(num_blocks, block_size, num_kv_heads, index_head_dim)` |
| `kv_cache[3]` | indexer key scale（仅 Sparse C8） | `(num_blocks, block_size, num_kv_heads, 1)` |

布局均为 **PA_BSND**（Page-Aligned, Block-Size-Token-NumHeads-HeadDim）。

### 2.2 寻址逻辑

1. `BlockTable.compute_slot_mapping` 将 token position 映射为 slot id：
   ```
   slot = physical_block_id * block_size + offset_in_block
   ```
2. `AscendSFAImpl.exec_kv` 调用 `npu_kv_rmsnorm_rope_cache`，按 `slot_mapping` 将当前 token 的 KV 写入 `kv_cache[0]` / `[1]`。
3. `forward` 中用 `npu_scatter_nd_update_` 将 indexer key `k_li` 写入 `kv_cache[2]` / `[3]`。
4. `indexer_select_post_process` 调用 `npu_lightning_indexer`，基于 `q_li` 和 `kv_cache[2]` 得到 `topk_indices`。
5. `_execute_sparse_flash_attention_process` 调用 `npu_sparse_flash_attention`：
   - `sparse_indices = topk_indices`（逻辑 token id，请求内从 0 开始）
   - `block_table = attn_metadata.block_table`
   - 算子内部用 `block_table` 把逻辑 token id 转成物理 slot。

### 2.3 关键接口

```python
# lightning indexer
npu_lightning_indexer(
    query=q_li,              # [num_tokens, num_heads, head_dim]
    key=kv_cache[2],         # [num_blocks, block_size, num_heads, head_dim]
    weights=weights,
    actual_seq_lengths_query=...,
    actual_seq_lengths_key=...,
    block_table=block_table,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=2048,
    sparse_mode=3,
) -> topk_indices            # [num_tokens, num_heads, topk]

# sparse flash attention
npu_sparse_flash_attention(
    query=ql_nope,
    key=kv_cache[0],
    value=kv_cache[0],
    sparse_indices=topk_indices,
    scale_value=...,
    sparse_block_size=1,
    block_table=block_table,
    actual_seq_lengths_query=...,
    actual_seq_lengths_kv=...,
    query_rope=q_pe,
    key_rope=kv_cache[1],
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
) -> attn_output
```

---

## 3. 方案 A：外挂非顺序映射缓存池（推荐用于验证）

### 3.1 核心思路

- **不改变原 KV Cache 的分配、写入和 layout**。
- 每个 request 的每个 attention layer 都有专门的外挂 cache；slot 分配粒度是 **per-layer per-token**。
- 新生成 KV 仍按 vLLM 原生 layout 写入原 KV Cache；只有 KV 被 offload 之后，才进入新的 cache 管理和映射体系。
- 写入和读取使用不同映射：
  - 写入当前 decode token：继续使用 vLLM 原生 `slot_mapping` / `block_table`。
  - 读取已 offload token：使用 `OffloadKVCacheLookup` 和新的 offload 映射。
- 新缓存池中的 KV **不按请求或 token 顺序排布**；逻辑 token 到新缓存 slot 的关系由每层的 `offload_slot_table` 和 lookup 统一管理。
- SFA 算子需要适配新的 block table layout；不能假设原生 `block_table` 的 layout/语义仍然成立。
- `indexer` 和 SFA 对新 layout 的具体需求需要先看算子实现后确定。
- 在 decode SFA forward 中：
  1. 当前 token 的 KV 先写入原始 `kv_cache`。
  2. offload 发生后，cache manager 将对应 KV 放入每层专属 `offload_kv_cache`，并更新 `offload_slot_table`。
  3. `indexer` 输出仍保持原有 top-K index 格式。
  4. `OffloadKVCacheLookup` 输出与 indexer 相同 shape/dtype 的 mapped indices。
  5. SFA 通过适配后的 offload block table layout 在新缓存池中寻址。

### 3.2 新缓存池的非顺序映射策略

新缓存池不采用“请求内连续 block”或“逻辑 token 顺序等于物理 slot 顺序”的假设。每层 `OffloadKVCachePool` 维护一张常驻 NPU 的映射表：

```
offload_slot_table[layer_id][req_idx, logical_token_id] = physical_slot_id
```

关键要求：

- `physical_slot_id` 是 layer 内 token slot，不要求连续、递增或与原 `block_table` 共享。
- 不同 layer 的 cache 和 slot table 相互独立，不能共享 slot id。
- 当前 decode token 的生成写入不走 `offload_slot_table`。
- token 被 offload 后，才由 cache manager 分配新 slot 并更新 `offload_slot_table`。
- lookup 只负责读路径映射：把 indexer top-K 输出映射成 SFA 可消费的 offload indices。
- 第一阶段可以先使用确定性的非顺序映射验证链路；具体 allocator / eviction / offload 触发策略待定。

### 3.3 新增/修改文件

| 文件 | 改动 |
|---|---|
| `vllm_ascend/attention/offload_kv_cache.py`（新建） | `OffloadKVCachePool`：按 request/layer 管理外挂 KV cache tensor 与 `offload_slot_table`；`OffloadKVCacheLookup`：读路径逻辑 index → offload index。 |
| `vllm_ascend/worker/model_runner_v1.py` | 在 `initialize_kv_cache_tensors` 中为 SFA 模型创建 `OffloadKVCachePool`；在 `execute_model` 中传入 forward context。 |
| `vllm_ascend/ascend_forward_context.py` | `set_ascend_forward_context` 增加 `offload_kv_cache_pool` 参数并挂到 `forward_context`。 |
| `vllm_ascend/attention/sfa_v1.py` | decode 写入继续走原生 `slot_mapping`；在 offload 后更新外挂 cache；构造 `req_id_per_token` 并调用 lookup；`_execute_sparse_flash_attention_process` 适配新的 offload block table layout。 |
| SFA 算子实现 | 根据具体实现适配新的 block table layout；具体改动点需先阅读算子代码确认。 |

### 3.4 lookup 算子设计（必须在 NPU 上执行）

`OffloadKVCacheLookup` 是 `indexer → SFA` 之间的算子，**全程在 NPU 上运行**，CPU 不参与：

- `offload_slot_table` 创建后常驻 NPU，不会被拷贝到 CPU。
- `req_id_per_token` 必须在 NPU 上构造（例如通过 `cum_query_lens` 生成索引张量，再广播到每个 token）。
- lookup 内部仅使用 NPU 支持的 PyTorch 索引/算术操作（整数索引、`expand`、`clamp`、`masked_fill`）。
- lookup 的**输出格式必须与 indexer 输出保持一致**：shape/dtype 与 `topk_indices` 一致，即 `[num_tokens, num_heads, topk]`。
- lookup 输出中每个元素的具体语义（物理 slot id、block-table index 或其他 SFA address index）取决于 SFA 适配后的 block table layout，先待定。

伪代码：

```python
def lookup(topk_indices, req_id_per_token, offload_slot_table):
    # topk_indices:          [num_tokens, num_heads, topk], NPU int32
    # req_id_per_token:      [num_tokens], NPU int32
    # offload_slot_table:    per-layer [num_reqs, max_seq_len], NPU int32
    # 输出 mapped_indices:    与 topk_indices 相同 shape/dtype

    padding_mask = topk_indices < 0
    safe_indices = topk_indices.clamp(min=0)

    req_ids = req_id_per_token.view(-1, 1, 1).expand_as(safe_indices)
    mapped_indices = offload_slot_table[req_ids, safe_indices]
    mapped_indices = mapped_indices.masked_fill(padding_mask, -1)
    return mapped_indices
```

> 注：上述所有张量操作都应在 `topk_indices.device`（NPU）上执行，不触发 host-device 同步。

`req_id_per_token` 的 NPU 构造示例（避免 `.item()` 同步）：

```python
cum_query_lens = attn_metadata.cum_query_lens  # [num_reqs], NPU int32
num_tokens = attn_metadata.num_input_tokens
query_start_loc = torch.cat([
    torch.zeros(1, dtype=cum_query_lens.dtype, device=cum_query_lens.device),
    cum_query_lens,
])
token_indices = torch.arange(num_tokens, device=cum_query_lens.device, dtype=torch.int32)
# token_indices[i] 属于哪个 request：找到最大的 query_start_loc[j] <= i
req_id_per_token = (token_indices.unsqueeze(0) >= query_start_loc[:-1].unsqueeze(1)).sum(dim=0) - 1
```

### 3.5 无 NPU 调试

引入环境变量 `VLLM_ASCEND_OFFLOAD_KV_MOCK=1`，它是一条**与真实 NPU 路径严格隔离**的调试分支：

- `OffloadKVCachePool` 的 tensor 放在 CPU。
- `OffloadKVCacheLookup` 的 mock 版本在 CPU 上用普通 `torch` 索引实现，仅用于验证 decode 读路径的非顺序映射和输出格式。
- `_execute_sparse_flash_attention_process` 在 mock 模式下返回 `torch.zeros_like(ql_nope)`，仅验证 shape 与数据流。

真实运行路径（未设置 mock）下，所有 tensor 仍在 NPU 上，lookup 仍走 NPU 实现。

### 3.6 优缺点

| 优点 | 缺点 |
|---|---|
| 原生 decode KV 写入路径保留完整。 | offload 后需要额外存储外挂副本。 |
| 新缓存池从第一阶段起就具备独立、非顺序的 slot 语义。 | SFA 算子必须适配新的 block table layout。 |
| 容易在无 NPU 环境下调试验证。 | 后续真实 eviction / 压缩策略仍需单独设计。 |

---

## 4. 方案 B：连续 layout + 手动 gather attention（备选）

### 4.1 核心思路

- 外挂缓存池按请求、按层组织为**连续张量**：
  ```
  [num_reqs, max_seq_len, num_kv_heads, head_dim]
  ```
- 不再使用 PA_BSND block layout，也不再依赖 `block_table`。
- `lightning indexer` 仍输出逻辑 token id；lookup 直接输出每个 topk token 在全局连续空间中的**展平索引**。
- 该方案可以体现新张量 layout，但与“KV 不按顺序排布、由 lookup 统一映射”的当前设计方向不完全一致。
- SFA 不再调用 `npu_sparse_flash_attention`，而是：
  1. 用 `torch.gather` 或自定义算子从连续缓存池中收集 top-K 个 token 的 KV。
  2. 用标准 attention 算子（或新封装算子）计算。

### 4.2 优缺点

| 优点 | 缺点 |
|---|---|
| 真正体现"新的 KV Cache layout"，与 block-based paged cache 解耦。 | 需要替换核心的 `npu_sparse_flash_attention` 调用，实现复杂度高。 |
| 更容易实现按请求的动态压缩/驱逐策略。 | 与现有 CUDA graph、量化、Sparse C8 路径冲突风险大。 |
| | 无 NPU 时难以验证数值正确性，需要完整 PyTorch 参考实现。 |

---

## 5. 两个方案对比

| 维度 | 方案 A | 方案 B |
|---|---|---|
| 侵入性 | 低 | 高 |
| 是否替换原 KV Cache | 否，原 cache 保留，offload 后外挂副本 | 否，但 attention 路径完全替换 |
| 是否复用 `npu_sparse_flash_attention` | 复用主体，但需要适配 block table layout | 否 |
| 是否体现新 layout | 强（独立非顺序 slot 映射） | 强（连续张量 layout，但不符合当前非顺序排布方向） |
| 第一阶段范围 | 非 CP 路径 | 非 CP 路径，且 attention 路径完全替换 |
| 量化/scale | 第一阶段不支持 | 第一阶段不支持 |
| 无 NPU 可调试性 | 好 | 较差 |
| 推荐阶段 | **验证阶段首选** | 仅在必须替换 attention 路径时再评估 |

---

## 6. 已决策与待决策问题

当前已经收敛的设计决策：

1. **新缓存池的 layout 需要立即与原始 KV Cache 不同（已决策）**
   - 新缓存池中的 KV 不按请求或 token 顺序排布。
   - lookup 是逻辑 token id 到新缓存物理 slot 的统一映射入口。

2. **cache 粒度（已决策）**
   - 每个 request 的每个 attention layer 都有专门 cache。
   - slot 分配粒度是 per-layer per-token。

3. **新缓存池不与原 KV Cache 共享 block id（已决策）**
   - 不共享原 `block_table` / `slot_mapping`。
   - 不采用按请求连续分配作为目标形态。
   - 第一阶段先实现确定性的非顺序映射，后续再接入真实 allocator / eviction 策略。

4. **写入和读取映射分离（已决策）**
   - 新生成 KV 仍按 vLLM 原生 layout 写入。
   - 只有 offload 后才进入新的 cache 管理和映射。
   - lookup 只服务读取路径，不替代原生 decode 写入映射。

5. **lookup 输出格式（已决策）**
   - 输出 shape/dtype 与 indexer 输出保持一致。
   - 具体值语义由 SFA 适配后的 block table layout 决定，先待定。

6. **第一阶段只针对 decode（已决策）**
   - 不处理 prefill。

7. **CP 场景第一阶段不支持（已决策）**
   - 第一阶段只验证单卡/非 CP 路径。
   - 不修改 `vllm_ascend/attention/context_parallel/sfa_cp.py`。
   - 如果运行时检测到 CP 开启，必须显式禁用 `offload_kv_cache` 或提前报错。

8. **量化和 scale 第一阶段不支持（已决策）**
   - 暂不处理 Sparse C8 / indexer key scale。

剩余待决策问题：

1. **SFA 算子适配方式**
   - 需要查看 SFA 算子具体实现。
   - 预计必须适配新的 block table layout。
   - 具体输入字段、layout 和寻址语义待定。

2. **indexer 对新 layout 的需求**
   - indexer 是否继续基于原生 KV/indexer key 工作，还是也需要感知 offload layout，待定。

3. **lookup 输出值语义**
   - 输出格式已确定为跟 indexer 一致。
   - 输出值到底是 slot id、block table index 还是 SFA 内部 address index，待定。

4. **真实 allocator / eviction / offload 触发策略**
   - 第一阶段可先用确定性非顺序映射验证链路。
   - 真实策略待定。

5. **无 NPU 调试的预期深度**
   - 只验证 shape/数据流？
   - 还是需要一个 PyTorch 参考 attention 来验证数值？

---

## 7. 建议的推进节奏

1. **先确认 SFA / indexer 算子契约**：
   - 阅读 SFA 算子实现，明确新的 block table layout 需要怎么适配。
   - 确认 indexer 是否需要感知 offload layout。
   - 明确 lookup 输出值语义，但保持输出 shape/dtype 与 indexer 一致。
2. **实现 decode-only 最小链路**：
   - 新建 `offload_kv_cache.py`。
   - 在 `model_runner_v1.py` 创建 per-request/per-layer 外挂缓存池、`offload_slot_table` 并传入 forward context。
   - decode 当前 token 写入继续走 vLLM 原生 layout。
   - KV offload 后再进入新 cache 管理，并更新 per-layer `offload_slot_table`。
   - 在 `sfa_v1.py` 中通过 lookup 生成与 indexer 输出格式一致的 mapped indices。
   - 增加 CP、prefill、量化/scale guard。
3. **无 NPU 环境下用 mock 路径验证框架逻辑**。
4. **有 NPU 后验证 SFA 适配和数值正确性**。
5. **数值正确后再接入真实 allocator / eviction / offload 触发策略。**

---

## 8. 附录：关键代码位置速查

| 功能 | 文件 | 关键类/函数 |
|---|---|---|
| KV Cache 分配与 reshape | `vllm_ascend/worker/model_runner_v1.py` | `initialize_kv_cache_tensors`、`_allocate_kv_cache_tensors`、`_reshape_kv_cache_tensors` |
| Block table / slot mapping | `vllm_ascend/worker/block_table.py` | `BlockTable.compute_slot_mapping` |
| SFA attention 实现 | `vllm_ascend/attention/sfa_v1.py` | `AscendSFAImpl`、`exec_kv`、`indexer_select_post_process`、`_execute_sparse_flash_attention_process` |
| KV Cache spec 扩展 | `vllm_ascend/patch/platform/patch_kv_cache_interface.py` | `AscendMLAAttentionSpec` |
| Forward context 扩展 | `vllm_ascend/ascend_forward_context.py` | `set_ascend_forward_context` |
