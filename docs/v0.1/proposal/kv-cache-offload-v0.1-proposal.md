# v0.1 KV Cache Offload 算子化旁路验证方案

> 状态：Proposal
> 范围：vllm-ascend SFA eager 调试路径
> 目标：在不改变原 SFA 计算流的前提下，将 v0 的 Python 旁路校验逻辑下沉为 NPU 算子，验证 `lightning indexer -> lookup -> 旁路 cache` 的正确性

## 1. 背景

v0 版本已验证：**不带 NPU 算子的纯 Python 旁路校验版本在 eager + SFA backend 下可以跑通**。其路径为：

```text
prefill KV 写入 MicroKV
decode lightning indexer 产生 topk token_pos
Python 侧 lookup 将 token_pos 映射为旁路 cache slot
miss 时从 MicroKV 同步加载 KV
旁路 cache 读取结果与原 vLLM KV cache 寻址结果对比
```

v0.1 在此基础上进一步演进：
- 将 **lookup** 逻辑从 Python 下沉为 **AIV 算子**，在 NPU 上并行执行；
- 将 **旁路 cache 维护**（slot table 更新、eviction 标记）下沉为 **AICPU 算子**，处理控制流与外部状态交互；
- 验证结果仍作为旁路，不接入 `npu_sparse_flash_attention`；
- 索引结构仍只覆盖 **prefill 已经生成的 token**，decode 各 step 新产生的 token 仍不回写 MicroKV，因此无法在新增索引结构中命中。

## 2. 目标

1. 验证 AIV `lookup` 算子能否正确将 `topk_indices` 转换为旁路 cache 的 `offload_slot_id`。
2. 验证 AICPU `maintain` 算子能否正确更新 `offload_slot_table`、标记 eviction。
3. 保持 v0 的 Python 校验语义：比较结果仅用于断言/统计，不影响 SFA 输入。
4. 为后续 v0.2 将旁路 cache 真正接入 SFA 做技术储备。

## 3. 非目标

v0.1 不实现：
- 旁路 cache 接入 SFA 计算。
- decode 新 token 回写 MicroKV。
- 图捕获（CUDA/NPU graph）兼容。
- CP / Sparse C8 / 量化 KV cache 路径。
- 生产级 eviction 策略与异步 miss load。

## 4. 总体架构

```text
                    prefill
                      │
                      ▼
              ┌───────────────┐
              │  persist_microkv│  ← Python / AICPU 均可，v0.1 保留 Python 写入
              │   (Python)      │
              └───────────────┘
                      │
                      ▼
              lightning indexer
                      │
                      ▼
              ┌───────────────┐
              │  lookup (AIV) │  ← 新增 AIV 算子
              │               │
              │ 输入: topk_indices [T, H, K]
              │       offload_slot_table [128K]
              │ 输出: bypass_slot_ids [T, H, K]
              │       miss_mask [T, H, K]
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ miss load &   │  ← Python 同步加载（同 v0）
              │ table update  │
              │ (Python)      │
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ maintain      │  ← 新增 AICPU 算子
              │ (AICPU)       │
              │ 输入: new_slot_table_updates
              │ 输出: updated offload_slot_table
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ 旁路读取 &    │  ← Python 校验（同 v0）
              │ 比较          │
              └───────────────┘
                      │
                      ▼
              npu_sparse_flash_attention
              （原始 topk_indices / 原始 block_table / 原始 kv_cache）
```

## 5. 新增算子设计

### 5.1 lookup 算子（AIV）

**定位**：将 lightning indexer 输出的 `topk_indices`（token_pos）映射为旁路 cache 的 slot id，同时输出 miss 掩码。

**输入**：

| 名称 | Shape | Dtype | 说明 |
|---|---|---|---|
| `topk_indices` | `[num_decode_tokens, num_heads, topk]` | int32 | lightning indexer 输出 |
| `offload_slot_table` | `[slot_table_size]` | int32 | 旁路 cache 索引表，`-1` 表示未命中 |
| `prefill_lens` | `[num_reqs]` | int32 | 每个请求的 prefill token 数，用于过滤 |
| `token_req_indices` | `[num_decode_tokens]` | int32 | 每个 decode token 所属的 req index |

**输出**：

| 名称 | Shape | Dtype | 说明 |
|---|---|---|---|
| `bypass_slot_ids` | `[num_decode_tokens, num_heads, topk]` | int32 | 旁路 cache slot id；`-1` 表示 miss 或越界 |
| `miss_mask` | `[num_decode_tokens, num_heads, topk]` | bool / uint8 | `1` 表示需要 miss load |

**处理逻辑（每个 AIV 核处理一个 topk item）**：

```cpp
for each (decode_token, head, topk_rank):
    token_pos = topk_indices[decode_token, head, topk_rank]
    req_idx   = token_req_indices[decode_token]

    if token_pos < 0:
        bypass_slot_ids = -1
        miss_mask = 0
    elif token_pos >= prefill_lens[req_idx]:
        bypass_slot_ids = -1
        miss_mask = 0
    else:
        slot_id = offload_slot_table[token_pos]
        bypass_slot_ids = slot_id
        miss_mask = (slot_id == -1) ? 1 : 0
```

**为什么放在 AIV**：
- 该算子是大规模并行查表，适合 AIV 的 SIMD 能力。
- 无复杂控制流，只涉及整数比较与索引读取。

### 5.2 maintain 算子（AICPU）

**定位**：根据 Python 侧 miss load 的结果，更新旁路 cache 的 `offload_slot_table`，处理 eviction 标记。

**输入**：

| 名称 | Shape | Dtype | 说明 |
|---|---|---|---|
| `slot_table` | `[slot_table_size]` | int32 | 当前旁路 cache 索引表 |
| `update_positions` | `[N]` | int32 | 需要更新的 token_pos 列表 |
| `update_slot_ids` | `[N]` | int32 | 对应位置的新 slot id |
| `evict_positions` | `[M]` | int32 | 被 eviction 替换掉的旧 token_pos（可为空） |

**输出**：

| 名称 | Shape | Dtype | 说明 |
|---|---|---|---|
| `updated_slot_table` | `[slot_table_size]` | int32 | 更新后的索引表 |

**处理逻辑**：

```cpp
// 先失效被 evict 的旧位置
for i in range(M):
    old_pos = evict_positions[i]
    slot_table[old_pos] = -1

// 再写入新位置
for i in range(N):
    pos = update_positions[i]
    sid = update_slot_ids[i]
    slot_table[pos] = sid
```

**为什么放在 AICPU**：
- 维护操作涉及 eviction 决策、外部状态（MicroKV）交互，Python 侧已经决定好更新内容，AICPU 只需执行原子化写入。
- 不需要 Tiling，适合 AICPU 的灵活控制流。
- 可作为后续异步 miss load / cache manager 下沉的中间态。

## 6. 数据所有权（继承 v0）

### 6.1 MicroKV 记录

与 v0 保持一致：

```text
(req_id, layer_id, token_pos) -> MLA token record
```

record 内包含 `magic`、`version`、`dtype`、`k_nope_shape`、`k_pe_shape`、`payload_checksum` 等元信息。

### 6.2 旁路 cache

与 v0 保持一致：

```text
(req_id, layer_id)
  offload_slot_table: int32[128K]
  offload_k_nope_cache: tensor[capacity, num_kv_heads, kv_lora_rank]
  offload_k_pe_cache: tensor[capacity, num_kv_heads, qk_rope_head_dim]
```

### 6.3 索引覆盖范围

**关键约束**：v0.1 新增的 AIV lookup / AICPU maintain 索引结构，**只管理 prefill 中已经计算过的 token**。

原因：
1. decode 各 step 中新产生的 token 对应的 KV cache，在 v0 中就不写回 MicroKV。
2. 因此这些新 token 在 `offload_slot_table` 中不存在有效 slot，lookup 必然 miss。
3. v0.1 保持这一语义，避免提前引入 decode 回写逻辑，降低验证复杂度。

影响：
- decode 时，如果 `topk_indices` 指向 decode 新 token 位置（`token_pos >= prefill_len`），AIV lookup 直接跳过，不会产生误校验。
- 该限制会在 v0.2（或更高版本）引入 decode 回写后解除。

## 7. 与原计算流的关系

与 v0 相同，v0.1 仍不得改变原 SFA 计算流：

1. 不修改传给 SFA 的 `topk_indices`。
2. 不修改传给 SFA 的 `block_table`。
3. 不修改原始 `kv_cache[0]`、`kv_cache[1]`、`kv_cache[2]`。
4. 不把旁路 cache 作为 SFA key/value 输入。
5. 校验失败通过显式异常或日志中断调试，不产生错误 attention 输出。

## 8. 插入点

在 `vllm_ascend/attention/sfa_v1.py` 的 `AscendSFAImpl.forward()` 中，v0.1 的算子调用顺序为：

```text
exec_kv() 或 _sfa_preprocess_with_mlapo() 写入 kv_cache[0] / kv_cache[1]
npu_scatter_nd_update_() 写入 kv_cache[2]
if offload_v0_enabled and 含 prefill:
    persist_prefill_kv_to_microkv(...)        # Python，同 v0
topk_indices = indexer_select_post_process(...)

if offload_v0_enabled:
    bypass_slot_ids, miss_mask = lookup_aiv(   # 新增 AIV 算子
        topk_indices,
        offload_slot_table,
        prefill_lens_cpu,
        token_req_indices_cpu,
    )
    # Python 侧根据 miss_mask 同步加载 MicroKV
    # 更新 slot_table 后调用 maintain_aicpu(...)  # 新增 AICPU 算子
    # 旁路读取 & 比较（同 v0）

attn_output = npu_sparse_flash_attention(... original topk_indices ...)
```

## 9. 开关与 guard

沿用 v0 的环境变量：

```text
VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE=1
MICROKV_SOCKET=/tmp/microkv.sock
VLLM_ASCEND_KV_OFFLOAD_V0_CAPACITY=4096
```

启用后必须检查：

| 条件 | v0.1 行为 |
|---|---|
| CUDA/NPU graph capture 开启 | 报错或禁用 |
| CP / DSA CP 开启 | 报错或禁用 |
| Sparse C8 indexer 开启 | 报错或禁用 |
| indexer scale cache 存在 | 报错或禁用（v0.1 需显式检查 `kv_cache[3]`） |
| MicroKV 不可连接 | 报错或禁用 |
| 非 SFA attention backend | 不启用 |

## 10. 组件划分

### 10.1 vllm-ascend 侧

| 文件 | 修改目的 |
|---|---|
| `vllm_ascend/attention/offload_kv_cache_v0.py` | 保留 Python 校验骨架；新增 AIV / AICPU 算子调用封装；miss load 仍在此调度。 |
| `vllm_ascend/attention/sfa_v1.py` | 保留两个最小插入点，调用 `lookup_aiv` 与 `maintain_aicpu`。 |
| `csrc/lookup_offload_slot/` | 新增 AIV 算子工程：`_def.cpp`、`_proto.cpp`、`_tiling.cpp/.h`、kernel 实现、CMakeLists.txt。 |
| `csrc/maintain_offload_slot/` | 新增 AICPU 算子工程：CPU kernel `.cc`、op_proto、info cfg、CMakeLists.txt。 |

### 10.2 MicroKV 侧

v0.1 不改 MicroKV 协议，仅复用 v0 中已增加的 `KV_MLA_TOKEN` 语义。

## 11. 实现步骤

1. **AIV lookup 算子**：
   - 在 `csrc/` 下新建 `lookup_offload_slot/` 目录。
   - 实现 OpDef、InferShape、InferDataType、Tiling、kernel。
   - 在 `csrc/CMakeLists.txt` 中注册 op 目录（现有 `op_add_subdirectory` 机制自动发现）。
   - 编译后在 Python 侧通过 `torch.ops._C_ascend.lookup_offload_slot(...)` 调用。

2. **AICPU maintain 算子**：
   - 在 `csrc/` 下新建 `maintain_offload_slot/` 目录。
   - 实现 CPU kernel、op_proto、info cfg。
   - 由于当前 `csrc/` 主要按 AIV 结构组织，需要评估是否引入独立的 `cpukernel/` 构建路径，或在现有 `op_host` 中扩展 AICPU 支持。

3. **Python 侧封装**：
   - 在 `offload_kv_cache_v0.py` 中新增 `lookup_and_validate_with_ops()`。
   - 先调用 AIV `lookup`，拿到 `miss_mask`；再 Python 同步 miss load；再调用 AICPU `maintain`。
   - 最后执行与 v0 相同的旁路读取比较。

4. **单元测试**：
   - AIV lookup：构造 fake `topk_indices` 与 `slot_table`，验证输出 `bypass_slot_ids` / `miss_mask`。
   - AICPU maintain：构造 slot table 更新，验证 eviction 后旧位置被置 `-1`、新位置写入正确。
   - 端到端：与 v0 相同场景下比较，确保算子结果与 Python 实现一致。

## 12. AIV 与 AICPU 编译差异总结

| 维度 | AIV 算子 | AICPU 算子 |
|---|---|---|
| 执行位置 | AI Core 向量核 | 昇腾芯片 CPU |
| 编程接口 | Ascend C | 标准 C++ |
| 目录结构 | `op_host/` + `op_kernel/` | `op_host/` + `op_kernel/`（或独立 `cpukernel/`） |
| 必须文件 | `_def.cpp`、`_proto.cpp`、`_tiling.cpp/.h`、kernel `.cpp` | CPU kernel `.cc`、op proto、info cfg `.ini` |
| 注册宏 | `OP_ADD(MyOp)`、`REGISTER_TILING_DATA_CLASS` | `REGISTER_CPU_KERNEL(OP_TYPE, ClassName)` |
| Tiling | 必须有 | 不需要 |
| 编译器 | cce（AI Core 编译器） | `aarch64-target-linux-gnu-gcc`（交叉编译） |
| 产物 | `.o`/`.bin` + `cust_opmaster`/`cust_opapi` | `libcust_aicpu_kernels.so` + `cust_aicpu_kernel.json` |
| 调用方式 | `torch.ops._C_ascend.my_op(...)` | `torch.ops._C_ascend.my_op(...)` 或 aclnn |

## 13. 风险与待确认点

1. **AICPU 算子如何接入现有 `csrc/`**：当前 vllm-ascend `csrc/` 全是 AIV 结构，需要确认是否已有 AICPU 编译链路，或需要新增 `cpukernel/` 构建。
2. **`offload_slot_table` 的 NPU 存储位置**：AIV lookup 需要从 NPU 内存读取 `slot_table`，因此该表必须常驻 NPU；miss load 后需要与 AICPU maintain 同步更新。
3. **算子输入中 `prefill_lens` / `token_req_indices` 的同步**：这些 CPU tensor 需要 H2D 或作为 AIV 算子的 CPU 输入，需确认最小开销路径。
4. **算子注册名冲突**：需避免与已有 vllm-ascend 自定义算子或 CANN 内置算子重名。
5. **v0.1 仍依赖 eager**：AIV / AICPU 算子在图捕获下可能需要 fake kernel 或显式禁用。

## 14. 后续演进

v0.1 验证通过后，下一阶段考虑：

1. 将 AIV lookup 输出真正接入 SFA，替换原 `topk_indices` 中的部分项。
2. 引入 decode 新 token 回写 MicroKV 与旁路 cache。
3. 将 miss load 异步化，减少同步 D2H/H2D 开销。
4. 支持图捕获、CP、量化与 Sparse C8。
