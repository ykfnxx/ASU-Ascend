# vLLM-Ascend DSA Sparse 模块深度分析

## 1. 分析范围

- 源码仓库：`vllm019-DSA-offload`
- 源码分支：`tmp-opt`
- 源码 HEAD：`ad428d7bf6e481a8be2141c94dc35ffd31bcbfae`
- 核心目录：`vllm_ascend/dsa_sparse/`
- 模块数量：18
- 核心代码量：5,782 行

本文从控制面状态机、HBM/DRAM 内存分层、request/forward/layer 数据结构、tensor layout、逐层计算、ACL Graph、资源释放和性能风险等方面分析整个 `dsa_sparse` 包。Scheduler、ModelRunner、SFA 和 C++ GS 算子是直接上下游，本文只分析其与 `dsa_sparse` 的接口边界。

## 2. 核心结论

`dsa_sparse` 是一个把“完整上下文选择”和“稀疏上下文计算”解耦的运行时：

- Indexer KV 在 HBM 中保留完整上下文，供每步 lightning-indexer 计算 topK。
- MLA KV 的完整满块卸载到 worker-local DRAM hot store。
- HBM MLA resident plane 只保留固定 sparse budget 和正在增长的尾块。
- Gather-Selection 根据 topK 和上一轮 resident status 判断 hit/miss，将 miss KV 从 DRAM 换入 HBM，并输出 SFA 使用的 resident logical indices。

它同时解决三个问题：

1. MLA/full KV 不再随上下文长度线性常驻 HBM。
2. SFA 只计算 sparse budget 对应的 token。
3. continuous batching、dense/sparse mixed batch 和 graph replay 使用统一 row-mode 路径。

## 3. 系统边界

```text
Scheduler / KV block manager
  | ReqStage、resident len、budget、HBM block ids、block hashes
  v
NPUModelRunner.build_dsa_meta
  |
  +-- DSAModelForwardMeta             request 行账本
  +-- DSAForwardSparseDecodeBatch     forward 级 GS tensor
  +-- DSAForwardLayerBatch            layer dump/guard 计划
  v
DSASparseV1
  |
  +-- DSAResidentTokenPool            HBM resident metadata
  +-- DSAHotKVStore                   DRAM hot KV 和逻辑块表
  +-- DSALayerCacheRegistry           layer -> cache tensor 绑定
  +-- DSAGraphBuffersMixin            graph-stable tensor
  |
  +-- attention_begin
  +-- lightning-indexer
  +-- after_indexer -> Gather-Selection
  +-- SFA
  +-- attention_finished -> full block dump
```

`DSASparseV1` 位于 `vllm_ascend/dsa_sparse/dsa_sparse.py:142`，同时用于：

- Scheduler 进程：stage 规划和 HBM slot allocation 包装。
- Worker 进程：metadata、DRAM store、resident pool、layer hook、GS 和 graph buffer。

角色由 `DSASparseRole.SCHEDULER/WORKER` 区分，定义于 `dsa_types.py:21`。

## 4. 模块职责

| 模块 | 主要职责 | 生命周期 |
| --- | --- | --- |
| `dsa_config.py` | 解析 `additional_config["dsa_sparse_config"]` | 进程初始化 |
| `dsa_types.py` | `ReqStage`、`DSADecodeRowMode`、`INVALID_SLOT` | 全局稳定类型 |
| `dsa_spec_utils.py` | 判断 Indexer/MLA resident KV spec | KV 初始化/调度 |
| `dsa_sparse.py` | stage、allocation、metadata、layer hook、GS 编排 | 请求/forward 主路径 |
| `dsa_req_meta.py` | 单请求 forward 计划和 sparse window | 单个 forward |
| `dsa_forward_batch.py` | ReqMeta 到 full-batch tensor 和 dump 计划 | 单个 forward |
| `dsa_attention_layout.py` | 提取 query layout 和 full block table | 单个 forward |
| `dsa_batch_tensor_utils.py` | padding、排序和 HBM/DRAM block table | 单个 forward |
| `dsa_resident_pool.py` | 请求 pool row、resident count、slot-token status | Worker/request |
| `dsa_hot_kv_store_core.py` | DRAM arena、逻辑块表、hash/refcount、dump | Worker/request |
| `dsa_ascend_hot_kv_store.py` | Ascend swapped-memory arena | Worker |
| `dsa_layer_cache_zones.py` | 发现并固定绑定每层 cache tensor | Worker |
| `dsa_ascend_ops_backend.py` | GS 输入规范化、设备校验和算子调用 | 每层 decode |
| `dsa_graph_gate.py` | row-mode decode graph 准入 | 每个 forward |
| `dsa_graph_buffers.py` | capture dummy batch、稳定 tensor、replay copy | Worker/graph |
| `dsa_model_runner_state.py` | worker request state、finish/preempt | 请求 |
| `dsa_trace.py` | trace point 和 rank/layer filter | 进程/调试 |
| `__init__.py` | 包标记 | 导入时 |

## 5. 控制面状态机

### 5.1 ReqStage

```text
PREFILL
  -> DENSE_DECODE
  -> ENTER_SPARSE_DECODE
  -> SPARSE_DECODE
```

| Stage | Cache 语义 |
| --- | --- |
| `PREFILL` | MLA 和 Indexer 按完整上下文写 cache |
| `DENSE_DECODE` | 已进入 decode，但 MLA 仍使用 dense/full layout |
| `ENTER_SPARSE_DECODE` | 首次把 MLA full layout 转换为 sparse resident layout |
| `SPARSE_DECODE` | resident layout 稳定，每步 topK -> GS -> SFA |

“新满块 dump”不是 stage，它可以发生在 PREFILL、DENSE_DECODE 或 SPARSE_DECODE 中。定义见 `dsa_types.py:42`。

### 5.2 Sparse threshold

`DSASparseBase` 初始化：

```text
hbm_sparse_budget_tokens = round_up(configured_budget, block_size)
enable_dsa_prompt_len = hbm_sparse_budget_tokens + block_size
```

默认配置：

```text
hbm_sparse_budget = 2048
block_size = 128
enable_dsa_prompt_len = 2176
```

当 `request.num_tokens <= 2176` 时继续 dense decode。超过阈值后：

```text
candidate_full_blocks = (total_tokens - 1) // block_size
tail_slots_need       = total_tokens - candidate_full_blocks * block_size
sparse_budget_tokens  = min(configured_budget,
                            candidate_full_blocks * block_size)
resident_valid_len    = sparse_budget_tokens + tail_slots_need
```

对应 `dsa_sparse.py:1052 plan_decode_resident_slots()` 和 `:1090 _plan_sparse_decode_resident_slots()`。

Resident plane 分为：

```text
[0, sparse_budget_tokens)       可被 GS 替换的 sparse budget
[resident_tail_start, end)      当前未满尾块，继续接受 decode 写入
```

### 5.3 RowMode

`DSADecodeRowMode` 是传给 GS/SFA 的逐行行为，不等同于 ReqStage：

| RowMode | 值 | 行为 |
| --- | ---: | --- |
| `PAD` | 0 | graph padding，不访问 cache |
| `DENSE` | 1 | 使用 native full-cache indices，不换入 KV |
| `SPARSE` | 2 | 执行 topK hit/miss、DRAM 换入和 resident indices |

这使 dense/sparse mixed batch 可以共用一个 GS 调用。

## 6. 内存分层

```text
HBM
  +-- Indexer dense plane：完整上下文
  +-- MLA resident plane：sparse budget + tail

Worker-local DRAM / swapped memory
  +-- NOPE_K arena[layer]
  +-- ROPE_K arena[layer]
  +-- logical block table[request_pool_row, logical_block]
  +-- ready table[layer][request_pool_row, logical_block]

NPU metadata
  +-- resident count[request_pool_row, layer]
  +-- slot status[layer, request_pool_row, 1, 1, budget+1]
  +-- forward row-mode tensors
  +-- graph-stable buffers

CPU metadata
  +-- request_id <-> pool row
  +-- hash <-> DRAM pool block
  +-- DRAM block refcount/free list
  +-- full_dump_done_by_pool[request_pool_row, layer]
```

### 6.1 HBM Indexer plane

Indexer cache 保留完整上下文，因为每步 topK 都要扫描全部历史。典型 layout：

```text
[num_indexer_blocks, block_size, 1, index_head_dim]
```

DeepSeek-V3.2 BF16、block 128、index head 128 时，每层每 page 为 32 KiB。它由 Scheduler/KV manager 分配，`dsa_sparse` 只识别 group 并保持 dense allocation。

### 6.2 HBM MLA resident plane

```text
nopek_cache_zone: [num_blocks, block_size, 1, kv_lora_rank]
ropek_cache_zone: [num_blocks, block_size, 1, qk_rope_head_dim]
```

典型维度为 512 和 64。`resolve_layer_cache_zones()` 在 `dsa_layer_cache_zones.py:146` 解析这些 tensor。

`DSALayerCacheRegistry` 首次绑定后校验 data pointer、shape、dtype 和 device。cache tensor 若在 worker 生命周期中被替换，resident metadata 会失效，因此代码直接报错。

### 6.3 DRAM hot plane

`DSAHotKVStore` 为每层维护 NOPE_K 和 ROPE_K 两个 `_ArenaPoolState`：

```text
hash_to_pool_idx
pool_idx_to_hash
pool_ref_counts
free_block_ids
arena
```

Ascend 实现通过 `torch_npu.empty_with_swapped_memory()` 分配 NPU 可寻址的 swapped-memory arena，见 `dsa_ascend_hot_kv_store.py:44`：

```text
NOPE arena[layer]: [capacity+1, block_size, 1, kv_lora_rank]
ROPE arena[layer]: [capacity+1, block_size, 1, qk_rope_head_dim]
```

block id 0 保留为 null/padding，真实 DRAM blocks 从 1 开始。

### 6.4 DRAM logical table

```text
dram_block_table[request_pool_idx, logical_block_idx] = dram_pool_idx
dram_block_ready[layer_id][request_pool_idx, logical_block_idx] = bool
```

Logical table 不带 layer 维，因为同一 logical block 在所有层使用相同 pool id；每层 arena 在该 pool id 存放本层 payload。Ready table 区分每层是否已经完成 dump。

CPU 表通过 version 号缓存对应 device tensor，只有版本变化时重新复制，见 `dsa_hot_kv_store_core.py:405` 和 `:447`。

### 6.5 Resident metadata pool

`DSAResidentTokenPool` 不保存实际 KV，只保存：

```text
request_id -> resident_pool_idx
cached_counts: [max_reqs, num_layers] int32
resident_slot_token_status:
  [num_layers, max_reqs, 1, 1, max_tokens+1] int32
```

```text
status[layer, pool_idx, 0, 0, resident_slot]
    = 当前 slot 保存的原始 token/segment id
```

最后一个额外位置由 GS 保存 selection actual sequence length。默认 61 层、256 请求、budget 2048 时，status tensor 约为 122 MiB NPU metadata。

### 6.6 Dump readiness

Worker 另有：

```text
full_dump_done_by_pool: [max_active_reqs, num_layers] bool, CPU
```

它保证请求在某层进入 sparse decode 前，该层完整历史块已经 dump 到 DRAM。当前 dump 从 Python 视角同步，因此它是 phase-order assertion；未来异步化需要改为 event/completion 驱动。

## 7. DRAM Block Dump 和复用

### 7.1 触发条件

1. Prefill 完成时 dump 已形成的完整 MLA blocks。
2. Decode 跨 block boundary 时 dump 刚完成的新满块。

不满尾块不卸载，因为后续 decode 仍会写入。

### 7.2 Dump 流程

`attention_finished()` 最终调用 `DSAHotKVStore.dump_layer_blocks_for_requests()`，位于 `dsa_hot_kv_store_core.py:485`：

1. 使用 HBM physical block ids 取出 noPE/ROPE blocks。
2. 将 request id 绑定到 resident pool row。
3. 查询 `(request_pool_idx, logical_block_idx)` 的现有 DRAM pool id。
4. 若没有，再按 block hash 查询可共享 block。
5. 若仍没有，从 free list 分配新 pool block。
6. NOPE 和 ROPE arena 使用相同 pool id。
7. 更新 logical block table。
8. 为 request 增加 NOPE/ROPE 引用计数。
9. ready 置 false，复制两个 payload，ready 再置 true。

### 7.3 Hash 和引用计数

相同 full-block hash 可以共享 DRAM block。引用关系为：

```text
(request_id, layer_id, BlockType) -> set[pool_idx]
```

请求结束时减少引用；归零后删除 hash 映射并把 pool id 放回有序 free list，payload 不清零，后续分配时覆盖。

## 8. 数据结构生命周期

### 8.1 Request：ReqMeta 和 ReqForwardPlan

`ReqMeta` 位于 `dsa_req_meta.py:147`，描述一个请求在当前 forward 的视图：

```text
request_id / index_in_batch
prompt/output/computed/scheduled token 数
ReqStage
resident_valid_seq_len / sparse budget
resident_pool_idx
HBM full block ids / full block hashes
query start / query length
dense query positions / resident query positions
DRAM store 引用
```

它每个 forward 重建，不是跨 step 持久对象。

`ReqForwardPlan` 派生本轮动作：

```text
是否 sparse decode
是否首次进入 sparse
是否产生新满块
需要 dump 的 logical block 范围
resident tail start
tail valid token count
budget slot count
```

`ReqSparseDecodeForwardPlan` 再提取 lightning-indexer/GS 所需的 `range_start`、`range_end`、query range 和 budget。

### 8.2 Model forward

`DSAModelForwardMeta` 是短生命周期 Python 容器：

```text
requests: list[ReqMeta]
full_block_table_tensor
```

`build_dsa_meta()` 每个 model forward 重建它。随后 `_build_forward_batches_from_dsa_meta()` 一次遍历生成：

1. `DSAForwardSparseDecodeBatch`：lightning-indexer -> GS -> SFA 数据。
2. `DSAForwardLayerBatch`：attention begin/finished 的 dump 和 guard 数据。

### 8.3 Layer

每层只构造轻量 view：

- `DSALayerRuntimeBatch`：layer id、cache zones、dump tables、guard rows。
- `DSALayerSparseDecodeBatch`：layer id、resident view、cache zones、DRAM store。

full-batch query、range 和 block table 不复制到 layer view，避免每层重复 `index_select` 和 tensor 构造。

## 9. Forward Tensor Layout

记号：

```text
B = 原始 model batch 行数
R = row-mode active decode 行数
S = R 中真正 sparse 的行数
P = resident pool 最大请求数
L = 模型层数
K = sparse budget/topK
Q = 当前每行最大 query token 数，主路径通常为 1
M = 最大 logical block 数
RB = resident HBM block table 宽度
W = SFA attention indices 宽度
```

### 9.1 DSAForwardSparseDecodeBatch

| 字段 | Layout | dtype | 语义 |
| --- | --- | --- | --- |
| `resident_pool_indices_tensor` | `[R]` | int32 | row -> persistent request pool row |
| `query_position_rows_tensor` | `[R,Q]` | int32 | query 在 dense/resident 空间的位置 |
| `tail_valid_token_counts_tensor` | `[R]` | int32 | tail 有效 token 数 |
| `resident_tail_starts_tensor` | `[R]` | int32 | tail 在 resident space 的起点 |
| `query_start_locs_tensor` | `[R]` | int32 | query 在扁平 input 中的起点 |
| `query_lens_tensor` | `[R]` | int32 | 每行 query token 数 |
| `query_last_token_indices_tensor` | `[R]` | int64 | 从 q/weights 选择最后 query token |
| `range_starts_tensor` | `[R]` | int32 | Indexer candidate 起点 |
| `range_ends_tensor` | `[R]` | int32 | Indexer candidate 终点 |
| `candidate_lens_tensor` | `[R]` | int32 | candidate 长度 |
| `budget_lengths_tensor` | `[R]` | int32 | 每行 sparse budget |
| `batch_hbm_block_table` | `[R,RB]` | int32 | MLA resident physical block ids |
| `dram_block_table` | `[P,M]` | int32 | pool row -> DRAM pool block |
| `batch_dram_block_table` | `[R,M]` | int32 | active rows 的 DRAM block table |
| `batch_row_indices_tensor` | `[R]` | int64 | DSA row -> 原始 model batch row |
| `row_modes_tensor` | `[R]` | int32 | PAD/DENSE/SPARSE |
| `active_local_row_indices_tensor` | `[R]` | int64 | DSA 小表 active row |
| `active_batch_row_indices_tensor` | `[R]` | int64 | 原 attention tensor active row |
| `sparse_row_mask_tensor` | `[R]` | bool | active rows 中哪些是真 sparse |
| `sparse_local_row_indices_tensor` | `[S]` | int64 | DSA 小表 sparse 子集 |
| `sparse_batch_row_indices_tensor` | `[S]` | int64 | 原 batch sparse 子集 |

### 9.2 Local row 和 batch row

```text
原始 batch = [prefill_req, dense_req, sparse_req]
DSA decode 小表 = [dense_req, sparse_req]

active_local = [0,1]
active_batch = [1,2]
sparse_local = [1]
sparse_batch = [2]
```

- local row 索引 DSA 自建 tensor，如 candidate lens。
- batch row 索引原始 attention tensor，如 `q_li`、weights 和 topK。

若用 local row 索引 `q_li`，会错误取到 prefill row。该约束记录于 `dsa_forward_batch.py:108-128`。

### 9.3 三种 Block Table

```text
Indexer block table [B,max_dense_blocks]
  完整上下文的 Indexer HBM physical block ids

batch_hbm_block_table [R,resident_blocks]
  MLA resident HBM physical block ids

batch_dram_block_table [R,max_logical_blocks]
  原序列 logical block -> DRAM pool block id
```

Indexer table 服务 topK；后两张表服务 GS，不能混用。

## 10. 完整计算流程

### 10.1 初始化

1. `dsa_config.py` 解析配置并附着动态 CacheConfig 属性。
2. Worker 创建 `DSASparseV1(WORKER)`。
3. 初始化 ResidentTokenPool、LayerCacheRegistry 和 full-dump readiness table。
4. KV cache tensor 分配后，`AscendDSAHotKVStore.initialize_hot_cache_from_kv_caches()` 根据真实 cache shape 预分配 arena。
5. `hot_num_blocks = indexer_num_blocks * hot_cpu_block_multiple`，默认 multiple 为 3。

### 10.2 Scheduler 规划和 HBM allocation

`plan_decode_resident_slots()`：

1. 判断 prefill 是否完成。
2. 排除 spec token、encoder input 等未支持路径。
3. 判断是否超过 sparse threshold。
4. 计算 budget、tail 和 resident length。
5. 写入 next stage、budget 和 resident length。

`dsa_alloc_slots_wrap()` 按 group 分配：

- Indexer group 始终按完整 dense sequence。
- MLA group 在 dense 阶段按完整序列，在 sparse 阶段按 budget + tail。
- ENTER_SPARSE_DECODE 释放旧 MLA full blocks，必要时保留未满尾块，再分配 sparse blocks。

### 10.3 Worker build_dsa_meta

`dsa_sparse.py:275 build_dsa_meta()`：

1. 解析 MLA/Indexer group id。
2. 提取 cumulative query lens、两套 query positions 和 full block table。
3. 校验 full-block hash 完整性。
4. 取得 MLA/Indexer block ids。
5. 获取或分配 stable resident pool row。
6. 将同一 pool row 绑定给 DRAM store。
7. 构造 ReqMeta。
8. 一次 tensor 化为 sparse batch 和 layer batch。

该阶段不计算 topK、不搬 KV、不更新 resident slot。

### 10.4 Prefill

Prefill 通过 native MLA/SFA 路径完整写 HBM cache。每层结束时：

1. `attention_finished()` 读取 `DSAFullBlockDumpTables`。
2. 把完整 MLA blocks 复制到该层 DRAM arena。
3. 更新 logical block table 和 ready table。
4. 最后 prefill chunk 完成后标记 `full_dump_done_by_pool[pool_idx,layer]`。

Scheduler phase barrier 保证 worker dump 完成后，后续 step 才缩减/回收 MLA full blocks。

### 10.5 Dense decode

- Stage 为 DENSE_DECODE，RowMode 为 DENSE。
- 图模式下 dense row 也可进入统一 lightning-indexer -> GS 路径。
- GS 对 DENSE row 不改 resident status，生成 native full-cache indices。

这避免为 pure dense、pure sparse 和 mixed batch 捕获不同 operator sequence。

### 10.6 ENTER_SPARSE_DECODE

1. Scheduler 设置 ENTER_SPARSE_DECODE。
2. allocation 释放 MLA full blocks，只保留所需 tail。
3. 分配 sparse budget resident blocks。
4. Worker 每层执行 GS，建立初始 slot-token status。
5. 该阶段有 cache layout 副作用，固定走 eager。

### 10.7 稳态 sparse decode

```text
attention_begin
  -> resolve/bind cache zones
  -> 验证该层 prefill dump ready

lightning-indexer
  -> q_li / k_li / weights
  -> 完整 Indexer HBM cache 上计算 topK

after_indexer
  -> 构造 layer sparse batch
  -> 取得 resident status 和 DRAM arenas
  -> Gather-Selection
       hit: 复用 resident slot
       miss: DRAM -> HBM resident slot
       原址更新 slot-token status
       输出 resident logical indices
  -> 提交 resident counts

SFA
  -> resident MLA cache + resident logical indices

attention_finished
  -> 若形成新满块，则 dump 到 DRAM
```

## 11. Gather-Selection 数据契约

`AscendDSAOpsBackend.gather_selection_update()` 位于 `dsa_ascend_ops_backend.py:221`。

| 输入 | Layout | 说明 |
| --- | --- | --- |
| `selection_topk_indices` | `[R,K]`/`[R,1,K]`/`[R,1,1,K]` | 原始 token ids |
| `req_pool_entries` | `[R]` | row -> resident pool row |
| `selection_kv_cache` | `[resident_blocks,block,512]` | HBM noPE cache |
| `selection_k_rope` | `[resident_blocks,block,64]` | HBM RoPE cache |
| `selection_block_table` | `[R,RB]` | resident HBM blocks |
| `resident_slot_token_status` | `[P,1,1,K+1]` | 持久 slot-token mapping |
| `full_kv_cache` | `[dram_blocks,block,512]` | DRAM noPE arena |
| `full_k_rope` | `[dram_blocks,block,64]` | DRAM RoPE arena |
| `full_block_table` | `[R,M]` | logical block -> DRAM block |
| `candidate_lens` | `[R]` | 合法 topK 范围 |
| `row_modes` | `[R]` | PAD/DENSE/SPARSE |
| `budget_lengths` | `[R]` | sparse budget |
| `tail_valid_token_counts` | `[R]` | tail 有效长度 |
| `resident_tail_starts` | `[R]` | tail logical start |
| `query_position_rows` | `[R,Q]` | query resident positions |

GS 有两类输出：

1. `resident_slot_token_status` 原址更新，供下一 step 判断 hit/miss。
2. `attention_indices [R,W]`，只供当前 SFA 使用。

SPARSE row 的 attention indices 是 resident logical slot id，不是原 token id。原 token 与 slot 的映射只存在于 status。

`_commit_gather_selection_resident_metadata()` 使用 `counts.index_copy_()` 更新 `[P,L]` resident count。Sparse row 写 budget length，Dense row 保留原 count。该路径避免 `.item()` 和动态 boolean indexing，可进入 FULL graph。

## 12. Graph 模式

### 12.1 Gate

`evaluate_dsa_row_mode_decode_graph()` 只允许稳定 single-token decode：

```text
总 token 数 == row 数
row 数属于 capture sizes
每个请求 scheduled_tokens == 1
stage 只能是 DENSE_DECODE 或 SPARSE_DECODE
不能发生 full-block dump
sparse budget 等于固定 configured budget
resident metadata 已 ready
```

Empty batch、capture miss、multi-token、ENTER_SPARSE_DECODE 和新满块边界属于预期 eager。

### 12.2 Stable buffers

`DSAGraphBuffersMixin` 按 `(graph_phase,row_count,device)` 缓存固定 shape 的 sparse batch。

Capture：

1. 创建 dummy request ids 和连续 pool rows。
2. 填合法 dummy block table、range、budget 和 RowMode。
3. 清空 dummy rows 的 resident status。
4. 暂存真实 forward batch 和 resident counts。
5. 用 dummy batch 执行 GS operator sequence。
6. 恢复 counts，再清空 dummy status，避免假状态泄漏。

Replay：

1. 校验真实 tensor 不超过 graph shape。
2. 将真实数据 `copy_` 到稳定地址。
3. 未使用区域填 PAD、0 或 -1。
4. index tensor padding 使用安全 row 0，有效性由 mask 控制，避免 NPU gather 对 -1 assert。

## 13. 请求状态同步和清理

### 13.1 SchedulerOutput 到 Worker

`dsa_model_runner_state.update_states()` 维护：

- `context_full_blk_hashes`
- split KV group block ids
- finish/preempt 生命周期
- streaming request metadata

`normalize_dsa_decode_block_ids()` 防止 sparse MLA block table 被错误补齐为 dense 长度，同时保证 Indexer group 保持完整 block ids。

### 13.2 Finish

```text
清理 full-dump readiness
释放 ResidentTokenPool row
清零所有层 resident count/status
释放 HotKVStore request table row
减少 DRAM block 引用
引用归零的 block 返回 free list
```

### 13.3 Preempt

抢占只清理 full-dump readiness 和 resident count/status，不释放 `request_id -> pool_idx`，使恢复请求复用稳定 row。最终 finish 必须执行，否则长期抢占请求会占用 `max_active_reqs`。

## 14. 容量公式

### 14.1 HBM resident payload

单请求、单层 resident MLA payload 近似：

```text
resident_tokens * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
```

默认最大 resident tokens 为 `2048 + 128 = 2176`，BF16 下：

```text
2176 * (512 + 64) * 2 ~= 2.39 MiB / request / layer
```

这是 block pool 容量需求公式，不表示每个请求永久独占；实际由 active request、block sharing 和 scheduler admission 决定。

### 14.2 Indexer payload

单层完整 Indexer cache：

```text
context_tokens * index_head_dim * dtype_bytes
```

它仍随上下文增长，是 DSA 节省 MLA HBM 后剩余的主要线性项。

### 14.3 DRAM arena

```text
hot_num_blocks = indexer_num_blocks * hot_cpu_block_multiple
hot_cpu_block_multiple = 3（默认）
```

每层分别分配 NOPE 和 ROPE arena，并多保留 block 0。

### 14.4 Metadata

```text
resident status: L * P * (K+1) * 4 bytes
resident counts: P * L * 4 bytes
full dump ready: P * L bytes
DRAM logical table: P * ceil(max_model_len/block_size) * element_bytes
DRAM ready tables: L * P * logical_blocks bytes
```

Graph buffers 还会按 capture size 保存一套固定 shape forward tensors。

## 15. 性能分析

### 15.1 Host 热点

- `build_dsa_meta()`：每 forward 遍历请求并解析 block/query metadata。
- `_build_forward_batches_from_dsa_meta()`：list 收集、排序和 tensor 化。
- `dump_layer_blocks_for_requests()`：hash/refcount/table 更新。
- graph replay 前多张 tensor 的 `copy_`。

已有优化：

- forward tensor 一次构建、所有 layer 复用。
- cache zones 使用 dense layer list，避免热路径反复发现。
- DRAM block table 使用 versioned device cache。
- layer id tensor、resident status 和 graph buffers 预分配。
- dense/sparse 不拆子 batch，减少 gather/merge。

### 15.2 Device 热点

- lightning-indexer 扫描完整 Indexer cache。
- GS 执行 status compare、hit compact、miss KV copy 和 status 更新。
- SFA 在 resident indices 上执行 sparse attention。

GS 延迟强依赖：

- topK miss rate。
- swapped-memory 带宽。
- candidate length。
- resident slot 替换量。
- RowMode 分布。
- operator tiling 和 stream 排布。

`gather_selection_stats` trace 可以统计 hit/miss，但会 D2H，只能用于诊断。

## 16. 正确性约束和风险

1. **固定 budget**：status pool 要求实际 topK 等于 `max_tokens`，当前不支持同 worker 动态 topK。
2. **Pool 上限**：active/preempted 请求超过 `max_active_reqs` 会耗尽 resident rows。
3. **同步依赖**：依赖 SchedulerOutput 和 worker forward 严格对齐，因此关闭 async scheduling。
4. **Dump 原子性**：ready 在两个 arena copy 完成后才置 true；异步化必须绑定 stream event。
5. **Cache pointer 稳定**：worker 生命周期中 cache tensor 变化会使 resident metadata 失效。
6. **Graph 副作用**：ENTER_SPARSE_DECODE 和 full-block dump 不能直接 replay。
7. **CPU/NPU 同步**：trace 和部分 readiness guard 使用 `.item()`，不能进入 graph 热路径。
8. **DRAM 扩容**：Ascend swapped-memory 扩容会重分配并复制旧 arena，应依靠预估避免运行时扩容。
9. **抢占生命周期**：preempt 不释放 pool row，依赖最终 finish 回收。
10. **功能边界**：prefix cache、spec decode、chunked prefill、P/D mixed、PCP 和 colder-tier transfer 尚未完整支持。

## 17. 关键设计取舍

### 17.1 保留完整 Indexer HBM cache

每步 topK 需要完整上下文。卸载 Indexer cache 会在每次 sparse attention 前恢复全量 selector KV，通常得不偿失。

### 17.2 只卸载 full blocks

只有 full block 有稳定 logical boundary 和 hash。尾块仍接受 decode 写入，留在 HBM 可避免 token 粒度的一致性协议。

### 17.3 Row-mode 而不是拆 batch

拆 dense/sparse 子 batch会增加 q/weights/topK gather、结果 merge、batch 顺序维护和 graph 组合数量。Row-mode 让一个 operator sequence 覆盖 PAD/DENSE/SPARSE。

### 17.4 Status 属于 ResidentTokenPool

Status 是跨 step、按 request/layer 生命周期存在的权威状态，不是无状态 backend 的临时输出，所以由 resident pool 持有，并在 finish/preempt 时统一清理。

### 17.5 DRAM logical table 不带 layer 维

同一 request logical block 在所有 layer 使用相同 pool id，payload 分散在各层 arena。共享 logical table 减少 metadata，per-layer ready table 区分各层数据是否完成。

## 18. 推荐阅读顺序

1. `dsa_types.py`：ReqStage 和 RowMode。
2. `dsa_config.py`：budget、request pool 和 graph 配置。
3. `dsa_sparse.py:81-194`：manager persistent resources。
4. `dsa_sparse.py:1052-1135`：Scheduler stage/resident 规划。
5. `dsa_req_meta.py`：单请求 forward plan。
6. `dsa_forward_batch.py:99-161`：tensor layout 和 row mapping。
7. `dsa_sparse.py:275-410`：build_dsa_meta。
8. `dsa_resident_pool.py`：HBM resident metadata。
9. `dsa_hot_kv_store_core.py`：DRAM arena、logical table、hash/refcount。
10. `dsa_sparse.py:706-807`：layer hook。
11. `dsa_ascend_ops_backend.py:221-420`：GS 输入输出。
12. `dsa_graph_gate.py`、`dsa_graph_buffers.py`：graph contract。
13. `dsa_model_runner_state.py`：请求更新、finish 和 preempt。

## 19. 总结

`dsa_sparse` 是跨 Scheduler、Worker、Attention 和 NPU kernel 的稀疏 cache runtime：

```text
完整 Indexer HBM plane
        +
固定预算 MLA resident HBM plane
        +
worker-local DRAM full-block plane
        +
request/forward/layer 分层 metadata
        +
row-mode lightning-indexer -> GS -> SFA
        +
graph-stable persistent buffers
```

最关键的索引连接点是：

- Scheduler ReqStage 决定 HBM physical allocation。
- resident pool row 把 request 映射到 NPU 可索引整数空间。
- DRAM logical table 把原序列 logical block 映射到 hot-store pool block。
- resident status 把原 token id 映射到当前 HBM sparse slot。
- GS 同时更新 KV payload、持久 status 和当前 SFA indices。

这四种索引空间保持一致，continuous batching、mixed row-mode 和 graph replay 才能共享同一条计算路径。
