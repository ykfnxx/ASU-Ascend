# v0.1.1 KV Cache Offload Block Table 适配设计

> 状态：Design
> 范围：vllm-ascend SFA eager decode 路径
> 前置：v0.1 已接入 `asu_hbm_index_lookup` 和 AICPU `asu_hbm_index_maintain_aicpu`
> 目标：让 SFA 真实读取紧密排布后的驻留 HBM KV cache，而不是只做旁路校验

## 1. 设计结论

v0.1.1 从 v0.1 的旁路校验推进到 SFA 计算输入适配。核心结论：

1. `offload_k_nope` / `offload_k_pe` 不能由 offload manager 自行 `torch.empty` 分配，必须来自 vLLM 初始化 KV cache 时分配或预留的物理 block。
2. lightning indexer 仍使用原始 `kv_cache[2]` 和原始 `block_table` 产生原始 token 坐标系下的 `topk_indices`。
3. SFA 只要求 `kv_cache`、`block_table`、`sparse_indices`、`actual_seq_lengths_kv` 在同一个逻辑 KV 坐标系下自洽，不依赖原始 vLLM block id 的业务语义。
4. SFA 前增加一个坐标转换层：原始 `topk_indices(token_pos)` 经 HBM index lookup 转换为 compact `slot_id`，再传给 SFA。
5. SFA 使用 compact `block_table`，该表把 compact slot 地址空间映射到 vLLM 预留的 offload physical blocks。
6. 为了真正压缩 K/V 驻留量，K/V cache 的 compact block table 必须和 indexer 使用的原始 block table 解耦；不能继续假设 `kv_cache[0]`、`kv_cache[1]`、`kv_cache[2]` 在所有算子里共用同一套逻辑寻址。
7. v0.1.1 只覆盖 eager `DecodeOnly` 调试路径；混合 prefill、spec decode、CP / DSA CP、Sparse C8、图捕获仍不进入本阶段。

## 2. 背景

v0.1 的真实算子接入仍保持原始 SFA 计算流：

```text
lightning indexer -> topk_indices(token_pos)
asu_hbm_index_lookup / maintain -> 旁路 HBM index 和旁路 KV cache
旁路 KV cache 与原始 vLLM KV cache 做一致性比较
SFA 继续读取原始 kv_cache + 原始 block_table + 原始 topk_indices
```

这个路径可以验证真实 lookup / maintain 算子语义，但不能减少 SFA 读取侧的 HBM KV 驻留量。v0.1.1 的目标是让 SFA decode 真正改读 compact resident cache：

```text
原始 indexer topk token_pos
        |
        v
HBM index lookup: token_pos -> compact slot_id
        |
        +--> miss 从 MicroKV 加载 KV 到 vLLM 预留 offload blocks
        |
        v
compact topk_indices(slot_id) + compact block_table + 原 kv_cache tensor 的预留 block 区间
        |
        v
npu_sparse_flash_attention
```

## 3. SFA 对 block_table 的实际依赖

从当前 vLLM-Ascend 代码看，SFA 对 `block_table` 的依赖集中在 page attention 的 KV 寻址：

1. Python 层 `AscendSFAImpl._execute_sparse_flash_attention_process()` 仅把 `attn_metadata.block_table` 作为 `npu_sparse_flash_attention` 的输入传入。
2. SFA tiling 中 `block_table.shape[1]` 被用于计算 `s2Size = maxBlockNumPerBatch * blockSize`。
3. kernel 中 `DataCopyPA()` 通过 `blockTableGm[batch, logical_block]` 读取 physical block id，并换算 key/value 的 GM offset。
4. `actual_seq_lengths_kv` 用于 sparse mode 的有效 KV 长度边界检查。

因此，SFA 不要求传入原始 request 的 block table。它要求以下输入在同一坐标系内一致：

| 输入 | v0.1 原始路径 | v0.1.1 compact 路径 |
|---|---|---|
| `kv_cache[0]` | 原始 K-nope cache 全量物理 blocks | 同一个 vLLM tensor，但 SFA 只通过 compact block table 读取预留 offload blocks |
| `kv_cache[1]` | 原始 K-rope cache 全量物理 blocks | 同一个 vLLM tensor，但 SFA 只通过 compact block table 读取预留 offload blocks |
| `block_table` | `token_pos // block_size -> original physical block` | `slot_id // block_size -> offload physical block` |
| `topk_indices` | 原始 `token_pos` | compact `slot_id` |
| `actual_seq_lengths_kv` | 原始 sequence length | compact slot 地址空间上界 |

## 4. vLLM block 所有权

### 4.1 不再独立分配旁路 KV tensor

v0.1 当前的旁路 cache 形态是：

```text
state.k_nope_cache: tensor[SLOT_COUNT, num_kv_heads, kv_lora_rank]
state.k_pe_cache:   tensor[SLOT_COUNT, num_kv_heads, qk_rope_head_dim]
```

v0.1.1 必须改为：

```text
state.offload_blocks: vLLM physical block ids
state.k_nope_cache:   kv_cache[0] 中 offload_blocks 对应的视图或寻址结果
state.k_pe_cache:     kv_cache[1] 中 offload_blocks 对应的视图或寻址结果
```

offload manager 只管理 logical slot 和 physical block id 的映射，不拥有独立 HBM storage。

### 4.2 offload pinned block pool

v0.1.1 引入一个 vLLM-owned offload pinned block pool：

```text
block_size = vllm_config.cache_config.block_size
slot_count = 10 * 1024
compact_blocks_per_req = ceil(slot_count / block_size)
```

如果 `block_size = 128`：

```text
compact_blocks_per_req = ceil(10240 / 128) = 80
```

每个 active offload request 需要一组 pinned compact blocks。物理 block id 来自 vLLM KV cache 的同一个 block id 空间：

```text
offload physical blocks for req row r:
  [offload_base + r * compact_blocks_per_req,
   offload_base + (r + 1) * compact_blocks_per_req)
```

这些 block 必须对普通 vLLM block allocator 不可见，否则原始 KV 写入可能覆盖 compact cache。

v0.1.1 推荐先采用静态 carve-out：

1. KV cache tensor 仍按包含 offload pool 的总 block 数分配。
2. 普通 scheduler / block allocator 可用 block 数扣除 offload pool。
3. offload pool 的 physical block ids 固定分配给 offload manager。
4. request 结束后只清理 offload manager 的状态，pinned block 回到 offload pool，不回到普通 allocator。

后续可演进为动态 pin：通过 vLLM block manager 为 offload request 申请 pinned blocks，并在 request 生命周期结束后释放。

### 4.3 K/V cache 与 indexer cache 的 block 视图分离

当前 SFA 路径里：

```text
exec_kv 写 kv_cache[0] / kv_cache[1]
npu_scatter_nd_update_ 写 kv_cache[2]
lightning indexer 读 kv_cache[2] + 原始 block_table
SFA 读 kv_cache[0] / kv_cache[1] + block_table + topk_indices
```

v0.1.1 的关键是只替换 SFA 读侧的 K/V 坐标系，不替换 indexer 坐标系。因此目标形态应拆成两套视图：

| 视图 | 使用方 | KV tensor | block table / slot mapping |
|---|---|---|---|
| 原始 indexer 视图 | lightning indexer | `kv_cache[2]` | 原始 `block_table` / 原始 `slot_mapping` |
| compact SFA 视图 | `npu_sparse_flash_attention` | `kv_cache[0]` / `kv_cache[1]` 的 offload pinned blocks | compact `block_table` / compact `topk_indices(slot_id)` |

这意味着 v0.1.1 的实现不能只在现有 full `kv_cache[0]` / `kv_cache[1]` tensor 尾部取一段预留 block 后结束。那只能验证 SFA 坐标转换是否正确，不能证明 K/V 驻留量已经降低。

要达成压缩目标，需要进一步满足：

1. `kv_cache[0]` / `kv_cache[1]` 的可读 resident K/V 只依赖 compact offload blocks。
2. `kv_cache[2]` 可以继续保留原始 indexer key cache，用于 lightning indexer 产生原始 token 坐标。
3. `exec_kv` / prefill persist 路径需要区分 K/V 写入位置和 indexer key 写入位置。
4. 普通 vLLM block allocator 对 K/V compact pool 不可见，但 indexer 原始 block table 仍可按现有路径维护。

因此 v0.1.1 文档中的 compact SFA 设计分为两层：

1. 计算适配层：证明 SFA 能用 compact topk + compact block table 读取 vLLM-owned offload blocks。
2. 驻留压缩层：让 `kv_cache[0]` / `kv_cache[1]` 的 HBM block 预算按 compact pool 分配，而不是继续按原始最大上下文分配。

### 4.4 Block ownership model

v0.1.1 必须把 block ownership 做成显式状态，而不是只依赖调用约定。ownership 的粒度是 `(storage_domain, physical_block_id)`：

```text
storage_domain = KV_PAYLOAD | INDEXER_KEY
```

其中 `KV_PAYLOAD` 对应 `kv_cache[0]` / `kv_cache[1]` 的 K/V payload storage，`INDEXER_KEY` 对应 `kv_cache[2]` 的 lightning indexer key storage。同一个数字 block id 可以同时出现在两个 storage domain 中，但它们不是同一个 ownership 对象。

| Owner | Storage domain | Block 范围 | 管理者 | 可见对象 | 允许写入方 | 允许读取方 |
|---|---|---|---|---|---|---|
| `NORMAL_KV_BLOCK` | `KV_PAYLOAD` | 普通 vLLM K/V payload block id | vLLM scheduler / block allocator | 原始 K/V `block_table`、原始 K/V `slot_mapping` | 原始 `exec_kv` / prefill K/V 写入路径 | 原始 SFA 对比路径、非 offload attention backend |
| `OFFLOAD_KV_BLOCK` | `KV_PAYLOAD` | compact K/V resident block id | `OffloadBlockPool` | compact `block_table` | offload miss load / resident window 初始化 | compact SFA |
| `INDEXER_BLOCK` | `INDEXER_KEY` | indexer key cache block id | 原始 indexer cache 路径 | 原始 indexer `block_table`、原始 indexer `slot_mapping` | `npu_scatter_nd_update_` 写 `kv_cache[2]` | lightning indexer |

其中 `NORMAL_KV_BLOCK` 和 `INDEXER_BLOCK` 在当前 vLLM-Ascend 实现中可以共享同一组原始 logical token 坐标和相同的数字 block id，但它们位于不同 storage domain。`OFFLOAD_KV_BLOCK` 与 `NORMAL_KV_BLOCK` 同属 `KV_PAYLOAD`，必须在 K/V payload allocator 可见性上互斥。`OFFLOAD_KV_BLOCK` 的唯一公开入口是 compact block table。

block owner 的不变量：

1. 普通 K/V allocator 分配 request K/V blocks 时，只能返回 `NORMAL_KV_BLOCK`。
2. 原始 K/V `slot_mapping` 只能落到 `NORMAL_KV_BLOCK` 对应的 physical slot，不能落到 `OFFLOAD_KV_BLOCK`。
3. 原始 indexer `slot_mapping` 只能写 `INDEXER_KEY` domain 的 `INDEXER_BLOCK`。
4. compact block table 只能包含 `OFFLOAD_KV_BLOCK`，不能包含普通 allocator blocks。
5. offload miss load 写 `kv_cache[0]` / `kv_cache[1]` 时，目标 slot 必须由 `slot_id -> compact block table -> OFFLOAD_KV_BLOCK` 得到。
6. compact SFA 调用时，`block_table` 必须是 compact block table；原始 SFA / indexer 调用时，`block_table` 必须是原始 block table。
7. request 结束时，`OFFLOAD_KV_BLOCK` 只能回到 `OffloadBlockPool`，不能直接进入普通 K/V allocator free list。

owner 状态可以用一个 host-side registry 表达：

```text
block_owner: int8[num_storage_domains, total_blocks]
  storage_domain 0 = KV_PAYLOAD
    0 = NORMAL_KV_BLOCK
    1 = OFFLOAD_KV_BLOCK
  storage_domain 1 = INDEXER_KEY
    2 = INDEXER_BLOCK
```

计算适配层允许 `kv_cache[0]` / `kv_cache[1]` 仍按全量 block tensor 分配，但必须满足 owner registry 的读写隔离。驻留压缩层必须进一步把 K/V block 预算收敛到 compact pool，让 `kv_cache[0]` / `kv_cache[1]` 不再按原始最大上下文保留完整 resident block 预算。

### 4.5 所有权检查点

v0.1.1 的实现入口需要放置以下检查：

| 检查点 | 检查内容 |
|---|---|
| KV cache 初始化后 | `KV_PAYLOAD` domain 的 `offload_reserved_blocks` 已标记为 `OFFLOAD_KV_BLOCK`，普通 K/V allocator 可用 block 数已扣除 |
| 构造原始 metadata 后 | 原始 K/V `block_table` 和 `slot_mapping` 不包含 `KV_PAYLOAD:OFFLOAD_KV_BLOCK` |
| 构造 indexer metadata 后 | indexer `block_table` 和 `slot_mapping` 只解释为 `INDEXER_KEY:INDEXER_BLOCK` |
| 构造 compact metadata 后 | compact `block_table` 全部来自 `KV_PAYLOAD:OFFLOAD_KV_BLOCK` |
| 写入 offload slot 前 | `slot_id` 映射到的 physical block owner 是 `KV_PAYLOAD:OFFLOAD_KV_BLOCK` |
| 调用 compact SFA 前 | `topk_indices` 已从 `token_pos` 改写为 `slot_id`，`actual_seq_lengths_kv` 使用 compact 上界 |
| request 结束后 | request 持有的 offload blocks 回到 `OffloadBlockPool`，owner 仍是 `OFFLOAD_KV_BLOCK` |

这些检查不是容错分支。v0.1.1 调试路径中任一检查失败都应直接报错，避免把原始路径和 compact 路径混用。

## 5. compact 坐标系

### 5.1 slot 到 block 的关系

HBM index lookup 输出的 `slot_id` 是 compact token slot：

```text
slot_id in [0, SLOT_COUNT)
logical_block = slot_id // block_size
block_offset = slot_id % block_size
```

compact block table row：

```text
compact_block_table[req_row, logical_block] = offload_physical_block_id
```

SFA kernel 最终访问：

```text
physical_block = compact_block_table[req_row, slot_id // block_size]
physical_slot = physical_block * block_size + slot_id % block_size
```

这个公式与原始 page attention 公式完全一致，只是逻辑坐标从 `token_pos` 变成了 `slot_id`。

### 5.2 topk_indices 改写

lightning indexer 输出仍是原始 token 坐标：

```text
topk_indices[decode_token, head, rank] = token_pos
```

SFA 前转换为 compact 坐标：

```text
compact_topk_indices[decode_token, head, rank] = lookup(token_pos) -> slot_id
```

转换流程：

1. 从原始 `topk_indices` 收集本 request/layer 的 unique `token_pos`。
2. 校验 token 属于 prefill backing store；不满足则报错，v0.1.1 不处理 decode 新 token 回写。
3. 调用 `asu_hbm_index_lookup` 得到 `slot_out`。
4. 对 miss token 从 MicroKV 读取 `k_nope/k_pe`。
5. 把 KV 写入 `slot_id` 对应的 vLLM offload physical slot。
6. 用 `token_pos -> slot_id` 映射改写整张 `topk_indices`。
7. 调用 AICPU maintain 回补 free pool。

### 5.3 actual_seq_lengths_kv

compact 路径下，`actual_seq_lengths_kv` 不再表达原始 request sequence length。它表达 compact slot 地址空间的有效上界：

```text
compact_seq_len = compact_blocks_per_req * block_size
```

`block_size = 128` 且 `SLOT_COUNT = 10240` 时：

```text
compact_seq_len = 80 * 128 = 10240
```

原因：

1. SFA sparse mode 会用 `actual_seq_lengths_kv` 对 `topk_indices` 做边界判断。
2. compact `slot_id` 不保留原始时间顺序，不能再用原始 `seq_len` 作为上界。
3. 因果约束必须在改写前由原始 `topk_indices(token_pos)` 保证：只允许已经存在于原始 prefill / decode 可见范围内的 token 进入 lookup。

因此 v0.1.1 的 SFA compact 调用使用：

```text
sparse_indices = compact_topk_indices
block_table = compact_block_table
actual_seq_lengths_kv = compact_actual_seq_lengths_kv
```

其中 `compact_actual_seq_lengths_kv[req_row] = compact_seq_len`。

## 6. Decode 数据流

v0.1.1 decode-only 流程：

```text
1. 原始 vLLM 流程写当前 token KV：
   exec_kv / mla_preprocess 产出当前 token 的 K/V
   npu_scatter_nd_update_ 按原始 slot_mapping 写 kv_cache[2]，供 indexer 使用
   K/V 是否写入原始 kv_cache[0] / kv_cache[1] 取决于当前阶段：
     - 计算适配层可以继续写原始 K/V blocks，用于对比和回退验证
     - 驻留压缩层应只写 compact resident / offload blocks

2. lightning indexer：
   输入：原始 kv_cache[2] + 原始 block_table
   输出：原始 topk_indices(token_pos)

3. offload compact 准备：
   token_pos -> query_index
   asu_hbm_index_lookup -> slot_id
   MicroKV miss load -> 写入 vLLM offload physical blocks
   topk_indices(token_pos) -> compact_topk_indices(slot_id)
   构造 compact_block_table
   构造 compact_actual_seq_lengths_kv

4. SFA：
   输入：原始 kv_cache tensor
        compact_topk_indices
        compact_block_table
        compact_actual_seq_lengths_kv
   行为：通过 compact block table 只读取 offload pinned blocks

5. maintain：
   AICPU maintain 根据 last_query_slots 回补 free pool
```

注意：第 4 步传入的 `kv_cache` 可以仍是原始 layer 的 `kv_cache` tuple；关键是 SFA 的 `block_table` 指向 offload physical blocks，因此 kernel 不会读取普通 request blocks。

## 7. Prefill 和 resident window

prefill 仍负责把 backing store 建好：

1. 原始 prefill KV 先按 vLLM 正常路径写入原始 KV cache blocks。
2. v0.1.1 将 prefill token 的 `k_nope/k_pe` 写入 MicroKV。
3. 对 resident window 内 token，同步拷贝到 offload pinned blocks。

resident window 初始化沿用 v0.1：

```text
resident_count = min(prefill_len, RESIDENT_SLOT_COUNT, INDEX_SIZE)
for token_pos in [0, resident_count):
  slot_id = token_pos
  index[token_pos] = slot_id
  slot_to_index[slot_id] = token_pos
  copy original kv[token_pos] -> offload physical slot(slot_id)
```

v0.1.1 不要求 SFA prefill 改读 compact cache。prefill 计算和混合 prefill/decode batch 仍走原始路径。

## 8. 状态划分

### 8.1 per request / layer HBM index state

保持 v0.1 的状态粒度：

```text
(req_id, layer_id)
  index: int32[1, 128K]
  slot_to_index: int32[1, 10K]
  free_slots: int32[1, 2K]
  free_head: int32[1]
  query_index: int32[1, 2K]
  last_query_slots: int32[1, 2K]
```

### 8.2 per request pinned block state

新增 compact block 状态：

```text
(req_id)
  offload_physical_blocks: int32[compact_blocks_per_req]
  compact_block_table_row: int32[compact_blocks_per_req]
  compact_actual_seq_len: int32 = compact_blocks_per_req * block_size
```

`offload_physical_blocks` 可以被同一 request 的所有 layer 复用为 block id 地址空间，但每个 layer 的 `kv_cache[0]/[1]` 是独立 tensor，因此物理 block id 相同不代表跨 layer 共享 payload。

## 9. 与 vLLM 原始 block_table 的关系

v0.1.1 不修改普通 `BlockTable.compute_slot_mapping()` 的语义。原因：

1. `slot_mapping` 主要服务当前 token KV 写入。
2. SFA decode 历史 KV 读取由 `block_table + topk_indices` 决定。
3. 全局改写原始 block table 会影响 prefill、indexer、prefix cache、scheduler 和其他 attention backend。

因此适配边界放在 SFA 内部：

```text
原始 metadata.block_table 只给 indexer 使用
compact block_table 只给 offload SFA 调用使用
```

如果 offload compact 准备失败，v0.1.1 调试路径应 fail fast，不静默回退到原始 SFA 后继续产出可能混淆的结果。

## 10. 文件修改范围

vLLM-Ascend 侧建议修改：

| 文件 | 责任 |
|---|---|
| `vllm_ascend/attention/offload_kv_cache_v0.py` | 将旁路 cache tensor 改为 vLLM physical block 视图；新增 compact topk 改写、compact block table 构造、compact actual seq len 构造 |
| `vllm_ascend/attention/sfa_v1.py` | 在 indexer 后、SFA 前调用 compact 准备；SFA decode-only 时传入 compact inputs；逐步拆分 K/V 写入 slot mapping 与 indexer key 写入 slot mapping |
| `vllm_ascend/worker/model_runner_v1.py` | 初始化 offload manager 时传入 block size、offload pool 配置和 request metadata；在 KV cache 初始化后注册 offload physical block pool；为 K/V compact pool 和 indexer cache 保留独立 block 预算；维护 block owner registry |
| `vllm_ascend/worker/block_table.py` | 不改普通寻址语义；如需要，仅增加 offload compact block table helper，不影响默认 block table |
| `vllm_ascend/envs.py` | 增加 v0.1.1 开关和 offload pinned request 数配置 |
| `tests/ut/attention/test_offload_kv_cache_v0.py` | 覆盖 slot_id 到 compact block table、topk 改写、actual seq len、MicroKV load 写入 physical slot |
| `tests/ut/attention/test_sfa_v1.py` | 确认 indexer 使用原始 inputs，SFA 使用 compact inputs |
| `tests/ut/worker/test_offload_block_ownership.py` | 覆盖 ordinary block table 不包含 offload blocks、compact block table 只包含 offload blocks、request 释放不污染普通 allocator |

不建议修改：

| 路径 | 原因 |
|---|---|
| `csrc/sparse_flash_attention/*` | v0.1.1 目标是框架侧适配，SFA kernel 已支持通过 block table 做 page attention 寻址 |
| `ascend-ops/asu_hbm_index_lookup/*` | lookup 语义仍是 `token_pos -> slot_id` |
| `ascend-ops/asu_hbm_index_maintain/*` | AICore maintain 不是目标路径 |

## 11. 开关和配置

建议引入独立开关，避免和 v0.1 旁路校验混淆：

```text
VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1
VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS=<N>
MICROKV_SOCKET=/tmp/microkv.sock
```

容量关系：

```text
offload_reserved_blocks = max_pinned_reqs * compact_blocks_per_req
normal_allocator_blocks = total_vllm_blocks - offload_reserved_blocks
```

如果 `normal_allocator_blocks <= 0`，启动失败。

## 12. 错误处理

v0.1.1 是调试路径，不做静默防御：

1. lookup op 或 maintain op 不存在：启动或首次进入 offload path 时报错。
2. request 没有 pinned offload blocks：报错。
3. 有效 query 数超过 `QUERY_COUNT`：报错。
4. `token_pos >= INDEX_SIZE`：报错。
5. MicroKV miss：报错。
6. compact `slot_id` 超出 compact block table 地址空间：报错。
7. block owner 检查失败：报错。
8. 非 DecodeOnly、CP、DSA CP、Sparse C8、图捕获：报错。

## 13. 测试策略

本机没有 NPU，v0.1.1 本地验证以静态和纯 Python 单元测试为主：

1. compact block table 构造：
   - `SLOT_COUNT = 10240`
   - `block_size = 128`
   - 生成 80 个 logical blocks
   - block table row 内容等于 vLLM offload physical block ids
2. topk 改写：
   - 原始 `topk_indices(token_pos)` 经 fake lookup 改为 `slot_id`
   - 重复 token 只 lookup 一次
   - 改写后 shape / dtype 保持不变
3. compact actual seq len：
   - decode-only 路径使用 `compact_blocks_per_req * block_size`
   - 不再使用原始 `seq_lens`
4. physical slot 写入：
   - fake MicroKV record 解包后写入 `kv_cache[0]/[1]` 对应 physical slot
   - 不创建独立 offload cache tensor
5. SFA wiring：
   - indexer mock 接收到原始 block table
   - SFA mock 接收到 compact block table 和 compact topk
6. block ownership：
   - 普通 allocator 可用 blocks 不包含 offload reserved range
   - 原始 block table / slot mapping 不包含 `OFFLOAD_KV_BLOCK`
   - compact block table 全部来自 `OFFLOAD_KV_BLOCK`
   - request 结束后 offload blocks 仍归 `OffloadBlockPool`，不进入普通 free list

需要在 Ascend 环境补充 e2e：

1. compact SFA 输出与原始 SFA 输出在同一 topk 下对齐。
2. SFA kernel 实际读取 offload physical blocks。
3. 多 decode step 后 lookup / maintain 的 `index` 与 `slot_to_index` 双向关系稳定。
4. request 结束后 offload pinned blocks 可复用。

## 14. 后续演进

v0.1.1 跑通后再考虑：

1. 将 offload pinned block pool 从静态 carve-out 改为 vLLM block manager 动态 pin / release。
2. 支持 batched `req_num > 1` 的 lookup / maintain。
3. 支持混合 prefill/decode batch，通过拆分 SFA 调用分别处理原始 prefill 和 compact decode。
4. 支持 spec decode，多 query token 时需要重新确认 sparse mode 3 下 compact `actual_seq_lengths_kv` 的边界语义。
5. 让 lightning indexer 也读取 compact / offload-aware index cache，进一步降低 `kv_cache[2]` 驻留 HBM。
