# dsa-sparse-0.23-eager 相对 dsa-sparse-0.23 的完整差异分析

## 1. 分析范围

本文按 2026-07-29 的仓库状态重新生成，内容以当前最终代码为准，不沿用旧文档中的数据结构和流程描述。

| 项目 | 值 |
| --- | --- |
| 仓库 | `/home/solidyang/workspace/vllm-ascend` |
| 基线分支 | `dsa-sparse-0.23` |
| 基线提交 | `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` |
| 基线远程引用 | `work/dsa-sparse-0.23`，与本地基线提交一致 |
| 当前分支 | `dsa-sparse-0.23-eager` |
| 当前提交 | `74f00dddc7fd76411058acd1d798084c65dc05ef` |
| 当前远程引用 | `origin/dsa-sparse-0.23-eager`，与本地当前提交一致 |
| Merge base | `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` |
| 提交关系 | 当前分支相对基线线性领先 33 个提交，基线侧无额外提交 |
| 修改规模 | 65 个文件，`10815 insertions(+), 574 deletions(-)` |

用于复核差异的命令为：

```bash
git diff --stat dsa-sparse-0.23...dsa-sparse-0.23-eager
git diff --name-status dsa-sparse-0.23...dsa-sparse-0.23-eager
git log --reverse --oneline dsa-sparse-0.23..dsa-sparse-0.23-eager
```

本文是源码和接口审计。本次文档刷新没有重新执行 A5/CANN 构建、单算子测试或 P/D 端到端测试；文中会明确区分代码已经实现的路径、mock 路径和仍未实现的生产能力。

## 2. 当前分支实现了什么

当前分支实现的是 GLM-5 DSA Sparse 在 Ascend A5 上的 eager Decode 框架路径，核心结果如下：

1. 将原来组合在一个 SFA KV cache 描述中的 Main MLA KV 与 Indexer KV 拆开。
2. P 节点仍保留完整的 Main 与 Indexer cache；D 节点只把 Indexer cache 暴露给 scheduler。
3. D 节点的 Main KV 不进入 scheduler KV block pool，而由 worker 为每个 sparse layer 单独分配固定大小的 Hot Main Cache。
4. 请求进入 Decode 时获得稳定的 `request_index`。每个 step 通过 `req_pool_entries` 把当前 batch 行映射到稳定请求行，不再使用 `row_to_seat` 或二次 seat 映射。
5. 按模型的 `skip_topk` 关系建立 IndexCache cohort。一个 cohort 共享一份 lookup 状态和一次 lookup 结果，但 cohort 内每层仍有独立的 Main Hot Cache。
6. 接入真实的 Ascend 950 SIMT 自定义算子 `dsa_sparse_lookup_update`。该算子同时完成 lookup、miss 分配和 metadata maintain。
7. 每个 cohort 每个 Decode step 调用一次 lookup 算子；每个 sparse layer 每个 step 分别调用一次 I/O 接口和一次 Sparse Flash Attention。
8. I/O 接口仍使用 no-op mock，没有实现 Main 历史 payload 的真实加载和生产级 P/D Main 发布。
9. 增加单算子 correctness、benchmark、profile 工具，以及 P/D 端到端 probe，用来确认自定义算子和 Hot Cache SFA 路径是否实际执行。
10. 当前明确不支持图模式、投机推理、多 token Decode、D 侧 prefill/mixed batch、PP/PCP/DCP 和 sequence-parallel token padding。

当前实现不能描述为“每层都调用 lookup 算子”。准确描述是：

> Main Hot Cache、I/O 调用和 SFA 调用是逐层的；lookup 状态和 lookup 调用是逐 IndexCache cohort 的。

如果模型中每个 sparse layer 都是 `skip_topk=False`，每层会各自形成一个 cohort，此时表现为每层一次 lookup。若后续层为 `skip_topk=True`，这些 follower layer 复用最近 leader 的 Top-K 和 lookup 结果，不重复调用 lookup。

## 3. Cache、状态和执行所有权

### 3.1 P 节点与 D 节点的 cache 所有权

| 资源 | P 节点 | D 节点 |
| --- | --- | --- |
| Indexer KV cache | scheduler 管理 | scheduler 管理 |
| Main MLA 全量 cache | 保留基线分配，用于 prefill 和传输源 | 不由 scheduler 分配 |
| Main Hot Cache | 不使用 DSA Sparse Hot Cache runtime | worker 按 sparse layer 独立固定分配 |
| lookup metadata | 不使用 | worker 按 IndexCache cohort 固定分配 |
| Main 历史 miss I/O | 作为未来生产后端的数据源 | 当前是 no-op mock |

`DSASparseExternalMainSpecs` 只是 D worker 内部保存的 Main cache 格式描述，不是存储对象，也不向 scheduler 新增 Main cache tensor。

在 D 节点上：

1. `get_kv_cache_spec()` 把 Main 的 `AscendMLAAttentionSpec` 存入 `DSASparseExternalMainSpecs`。
2. 返回给 scheduler 的 KV spec 中只保留 `AscendSFAIndexerCacheSpec`。
3. worker 初始化自己的 KV group 副本时，把 Main spec 补回元数据，使 attention backend 能看到完整层格式。
4. `runner_only_attn_layers` 使这些 Main layer 跳过 scheduler tensor 分配和 reshape。
5. Main attention module 只绑定零 block placeholder，维持模块 ABI。
6. 真正供 Decode 使用的 Main payload tensor 来自每层独立的 `DSASparseLayerHotCache`。

### 3.2 IndexCache cohort 的含义

`DSASparseCohort` 不是 scheduler KV cache group，也不是共享 Main payload 的 cache。

cohort 的建立规则位于 `vllm_ascend/worker/model_runner_v1.py:725-820`：

- `skip_topk=False` 的 sparse MLA layer 创建新 cohort，并成为 leader；
- 连续的 `skip_topk=True` layer 加入最近的 cohort；
- 第一层不能是 follower；
- cohort 名称使用 leader layer 名。

cohort 共享：

- `DSASparseLookupState`；
- semantic Top-K token positions；
- 本 step 的 `slot_out`、`miss_out` 和最终 `attention_indices`；
- token position 到 local Hot slot 编号的映射。

cohort 不共享：

- 每层 Main Hot Cache tensor；
- 每层 I/O context、region 和 completion；
- 每层当前 token 的 Main KV 写入；
- 每层的 I/O 调用；
- 每层的 SFA 调用。

因此同一 token 在 cohort 内各层使用相同的 local slot 编号，但不同层该 slot 对应不同 tensor 地址和不同 Main KV payload。

### 3.3 请求索引

当前实现已删除旧设计中的 `CacheSeatLease`、`row_to_seat` 和 seat epoch。

`RequestIndexManager` 在请求 admission 时分配一个稳定的整数：

```text
request_id -> request_index, 0 <= request_index < max_num_seqs
```

每个 Decode step 中，框架根据当前 request 顺序生成：

```text
req_pool_entries[batch_row] = stable request_index
```

算子直接用 `req_pool_entries` 选择 `index`、`slot_to_index`、`free_slots` 和 `free_head` 的请求行。batch 行顺序可以改变，但 request state 和该请求对应的 Hot Cache 区域不需要搬移。

CPU 仍参与请求生命周期中的 `request_id -> request_index` 字典管理，以及每个 step 构造 `req_pool_entries`；算子内部不存在 request ID 字符串处理。

### 3.4 D 节点上的两种 block table

当前 step 同时使用两种含义不同的 block table：

| 名称 | 生成方 | 作用 |
| --- | --- | --- |
| `block_table` | scheduler attention metadata | 表示请求的逻辑 block 到 Decode physical block 的映射；当前用于计算 `write_global_slots`，并传给未来真实 I/O 算子定位请求 Main region |
| `hot_block_table` | DSA Sparse worker | 根据稳定 `request_index` 合成 Hot Cache block 行，传给 Sparse Flash Attention 解释 local `attention_indices` |

`hot_block_table` 不是 scheduler 管理的第二套 KV block table，也不对应第二个 scheduler block pool。它只是 worker 为固定 Hot Cache 构造的 SFA 寻址表。

## 4. 固定容量与内存排布

固定常量定义在 `vllm_ascend/dsa_sparse_constants.py`：

| 常量 | 值 | 含义 |
| --- | ---: | --- |
| `DSA_SPARSE_INDEX_CAPACITY` | 128K | 单请求可寻址的 semantic token position 范围 |
| `DSA_SPARSE_RESIDENT_SLOT_COUNT` | 8K | 每请求初始驻留的有效工作集大小 |
| `DSA_SPARSE_FREE_SLOT_COUNT` | 2K | 每次融合更新使用的 free/swap slot 数量 |
| `DSA_SPARSE_LOOKUP_SLOT_COUNT` | 10K | local slot 编号空间，8K 初始有效项加 2K free 项 |
| `DSA_SPARSE_QUERY_WIDTH` | 2K | 每请求一次 lookup 的固定 Top-K 宽度 |
| `DSA_SPARSE_FREE_HEAD_STRIDE` | 16 | 每请求 `free_head` 行宽；元素 0 和 1 保存事务状态及循环淘汰游标 |

每请求 Hot Cache 还保留一个 block 作为当前 logical tail：

```text
hot_stride = 10K + block_size
live_tail_start = 10K
```

在当前要求的 `block_size=128` 下：

```text
hot_blocks_per_request = (10240 + 128) / 128 = 81
hot rows per request    = 10368
```

每个 sparse layer 都分配自己的：

```text
[max_num_seqs * 81, 128, ...per-token Main row shape...]
```

其中：

- 非 SFA C8：两个 plane，分别保存 `kv_lora` 和 `k_rope`；
- SFA C8：一个 packed plane。

### 4.1 持久固定 HBM

`DSASparseFixedHBMBreakdown` 当前只统计：

1. 每层 Hot Main payload；
2. 每 cohort lookup state；
3. 生产 backend 未来提供的 auxiliary bytes，当前为 0。

每 cohort lookup state 的字节数为：

```text
max_num_seqs * (
    131072  # index
  + 10240   # slot_to_index
  + 2048    # free_slots
  + 16      # free_head
) * sizeof(int32)
```

每层 Hot payload 的字节数为：

```text
max_num_seqs * (10240 + block_size) * per-token Main bytes
```

worker 在自动 memory profiling 和显式 `kv_cache_memory_bytes` 两条路径中，都先从可用于 KV blocks 的预算里扣除这部分固定 HBM。

### 4.2 算子临时 workspace

lookup 算子每次调用还需要临时 workspace：

```text
req_num * (10240 + 256 + 4) * sizeof(int32)
```

此外还要加 CANN `PlatformAscendC::GetLibApiWorkSpaceSize()` 返回的系统 workspace。该空间由 tiling 报告给运行时，是算子调用期临时空间，不计入 `DSASparseFixedHBMBreakdown` 的持久固定 HBM。

## 5. 请求生命周期

### 5.1 已定义的通用 P/D 生命周期

`DSASparsePDLifecycle` 定义了双 ready 状态机：

```text
begin_handoff
    |
    |-- mark_main_region_ready
    `-- mark_indexer_ready
             |
             v
        dual-ready notification
             |
             v
          admit
             |
             v
request_id -> stable request_index
```

每次 handoff 带 generation。迟到的旧 generation completion 不会修改当前请求状态。

完成、抢占和 handoff abort 都必须：

1. 确认请求不在活跃 step 中；
2. 释放 Main region handle；
3. 清空所有 cohort 中该 `request_index` 对应的 lookup state；
4. 释放 `request_index`。

### 5.2 当前实际接入的是 mock 生命周期

当前 `model_runner_v1.py:_update_states()` 使用：

- `_MockDSASparseRequestRegionBackend`；
- `admit_mock_request()`；
- `retire_mock_request()`。

新请求和 resumed 请求会立即模拟 Main-ready 与 Indexer-ready，然后获得 `request_index`。finished、preempted 和 resumed 边界会先 retire 旧 generation；resumed 请求随后重新 admission。

通用 `DSASparsePDLifecycle` 尚未连接真实 Main backend publication completion。当前 P/D connector 仍主要负责基线 Indexer/KV control plane，不能把 mock admission 解释为生产 Main 数据已经就绪。

## 6. 单个 Decode step 的完整调用链

### 6.1 step 外层

`vllm_ascend/worker/model_runner_v1.py:2618-2625` 在模型 forward 外建立 DSA Sparse execution context：

```text
NPUModelRunner.execute_model
  -> _begin_dsa_sparse_eager_execution
  -> DSASparseEagerRuntime.begin_target_batch
  -> 为每个 cohort 创建 DSASparseEagerBatchContext
  -> 把一个 DSASparseEagerContextRouter 挂到各层 attention metadata
  -> with dsa_sparse_execution: model forward
```

`_begin_dsa_sparse_eager_execution()` 强制：

- `AscendAttentionState.DecodeOnly`；
- 每请求本 step 恰好 1 个 token；
- 无 padding；
- metadata 中没有旧 DSA Sparse context。

### 6.2 step metadata

`DSASparseEagerCoordinator.build_step_metadata()` 位于
`vllm_ascend/attention/dsa_sparse.py:495-598`，生成：

- `req_pool_entries`；
- 当前 token `query_positions`；
- `seq_lens`；
- scheduler `block_table`；
- 当前 logical tail 起点；
- 当前 token 的 source/global slot；
- 当前 token 的 Hot Cache destination slot；
- write valid mask；
- synthetic `hot_block_table`。

当前 token Hot destination 的公式为：

```text
request_index * hot_stride + live_tail_start + token_offset_in_block
```

### 6.3 每层 Main 写入与 Top-K

每个 sparse layer 进入 `AscendSFAImpl.forward()` 后：

1. 在 `vllm_ascend/attention/sfa_v1.py:1630-1656` 取得本层 Hot Cache 和本层 live-tail slot mapping。
2. 在 `sfa_v1.py:1680-1682` 把 Main write slot 替换为 Hot Cache destination。
3. 现有 MLA/SFA prolog 直接把当前 token 的 Main KV 写入本层 live-tail block。
4. 在 `sfa_v1.py:1990-1991` 调用 `submit_newest_write(layer_name)`，记录本层当前 token 已写入。
5. leader 执行 Indexer 并生成 semantic Top-K；`skip_topk=True` follower 从现有 Top-K buffer 取相同结果。

当前实现没有独立的 `prepare_newest` 自定义算子。当前 token Main KV 写入继续复用现有 SFA prolog，`submit_newest_write()` 只是生命周期顺序标记。

### 6.4 自定义 lookup 算子的准确调用位置

生产路径中真正调用自定义算子的代码是：

```text
vllm_ascend/attention/sfa_v1.py:2040
  DSASparseEagerContextRouter.run_layer_attention(...)

vllm_ascend/attention/dsa_sparse.py:982
  DSASparseEagerBatchContext.run_layer_attention(...)

vllm_ascend/attention/dsa_sparse.py:900-908
  如果本 cohort 尚未 lookup：
  leader -> DSASparseEagerCoordinator.prepare_lookup(...)

vllm_ascend/attention/dsa_sparse.py:675
  self.lookup_operator.lookup(...)

vllm_ascend/ops/dsa_sparse.py:33
  torch.ops._C_ascend.dsa_sparse_lookup_update(...)
```

`DSASparseEagerBatchContext.run_layer_attention()` 只有在
`step.lookup_complete == False` 时调用 `prepare_lookup()`，并要求当前 layer 必须是 cohort leader。lookup 完成后，cohort followers 直接复用 `step.lookup_output` 和 `step.attention_indices`。

因此调用次数是：

```text
lookup calls per Decode step = active target cohort count
```

不是：

```text
lookup calls per Decode step = sparse layer count
```

### 6.5 每层 I/O 与 SFA

lookup 完成后，每个 sparse layer 都会进入
`vllm_ascend/attention/dsa_sparse.py:710-768`：

1. 在 `dsa_sparse.py:736` 调用一次 `dsa_sparse_io(...)`；
2. I/O 使用本层的 `DSASparseLayerBinding` 和本层 Hot Cache planes；
3. 当前 mock 只校验 shape、dtype、device 和 contiguous，不搬 payload；
4. 在 `dsa_sparse.py:760` 调用本层 attention closure；
5. closure 在 `vllm_ascend/attention/sfa_v1.py:1548` 调用现有 Sparse Flash Attention；
6. SFA 输入为本层 Hot Main Cache、`[req_num, 1, 2048]` 的 local sparse indices，以及 synthetic Hot block table。

所以 cohort follower 虽然不重复 lookup，仍会独立执行本层 I/O 和本层 SFA。

### 6.6 step 结束

一个 cohort 只有在其所有 layer 都完成后才能 `finish_step()`。模型 forward 抛异常时，`DSASparseEagerExecution.__exit__()` 会 detach metadata 并 abort 未完成 context，避免把 step-local context 留到下一个 forward。

## 7. `dsa_sparse_lookup_update` 算子接口

### 7.1 Torch 接口

Torch schema 注册在 `csrc/torch_binding.cpp:2955-2971`：

```text
dsa_sparse_lookup_update(
    index,
    slot_to_index,
    free_slots,
    free_head,
    req_pool_entries,
    query_index,
    lookup_mask,
    req_num
) -> (slot_out, miss_out)
```

其中 7 个输入是 Tensor，`req_num` 是整数 attribute。

| 参数 | shape | dtype | 读写 | 作用 |
| --- | --- | --- | --- | --- |
| `index` | `[P, 131072]` | int32 | 原地读写 | semantic token position 到 local slot |
| `slot_to_index` | `[P, 10240]` | int32 | 原地读写 | local slot 到 semantic token position |
| `free_slots` | `[P, 2048]` | int32 | 原地读写 | 本轮 miss 分配使用的 free slot 列表 |
| `free_head` | `[P, 16]` | int32 | 原地读写 | 事务 head 和循环 victim cursor |
| `req_pool_entries` | `[R]` | int32 | 只读 | 当前 batch 行到稳定请求行的直接映射 |
| `query_index` | `[R, 2048]` | int32 | 只读 | semantic Top-K token positions |
| `lookup_mask` | `[R, 2048]` | int32 | 只读 | 1 表示进入 history lookup，0 表示 tail、padding 或无效项 |
| `req_num` | scalar | int64 | 只读 | 当前并发请求数 R |
| `slot_out` | `[R, 2048]` | int32 | 输出 | 每个 history query 对应的 local slot；无效项为 -1 |
| `miss_out` | `[R, 2048]` | int32 | 输出 | canonical miss 为 1，hit、duplicate follower 和无效项为 0 |

`P` 是 `max_num_seqs`，`R` 是当前 active request 数。

框架侧不再向算子传：

- `resolved_hot_indices`；
- 独立 `miss_count`；
- `row_to_seat`；
- `state_seat_epoch`；
- `lru_slots`；
- 独立 workspace tensor；
- 独立 maintain 输入输出。

最终 SFA 使用的 `attention_indices` 在框架中构造：

- history query 使用算子 `slot_out`；
- 当前 logical tail 使用 `10240 + token_offset_in_block`；
- 无效项保持 -1。

### 7.2 Host、ACLNN 和 kernel 接入

调用链为：

```text
torch.ops._C_ascend.dsa_sparse_lookup_update
  -> dsa_sparse_lookup_update_torch_adpt.h
  -> aclnnDsaSparseLookupUpdate
  -> CANN op definition / infer shape / tiling
  -> dsa_sparse_lookup_update Ascend C kernel
  -> DsaSparseLookupUpdateSimt
```

对应位置：

| 位置 | 作用 |
| --- | --- |
| `csrc/attention/dsa_sparse_lookup_update/dsa_sparse_lookup_update_torch_adpt.h` | Torch tensor shape/dtype/device/contiguous 校验，分配两个输出，调用 ACLNN |
| `op_host/dsa_sparse_lookup_update_def.cpp` | 注册 7 个 int32 Tensor 输入、2 个 int32 输出、`req_num` attribute 和 `ascend950` 配置 |
| `op_host/dsa_sparse_lookup_update_infershape.cpp` | 输出 shape 跟随 `query_index` |
| `op_host/dsa_sparse_lookup_update_tiling.cpp` | 校验固定 shape，设置 tiling key 0、blockDim 和 system+user workspace |
| `op_host/op_api/aclnn_dsa_sparse_lookup_update.*` | 暴露 ACLNN GetWorkspaceSize 和执行入口 |
| `op_kernel/dsa_sparse_lookup_update.cpp` | AIV-only kernel 入口；按 request 分配 AIV block |
| `op_kernel/arch35/dsa_sparse_lookup_update_simt.h` | 256-thread SIMT lookup、分配和融合 maintain |

### 7.3 SIMT 算子逻辑

每个 request 的 SIMT block 执行以下阶段：

1. 把本请求 `slot_out` 初始化为 -1、`miss_out` 初始化为 0，并清空临时 protected-slot bitmap。
2. 读取 `req_pool_entries[req_id]`，直接定位四个持久 state tensor 的请求行。
3. 对所有有效 history query 查 `index[token]`：
   - hit：输出已有 slot，并把 slot 标记为本轮 protected；
   - miss：在 `index[token]` 中写入与 query 位置相关的负 claim。
4. 相同 token 的重复 miss 通过 deterministic claim 选出一个 canonical owner。
5. 统计 canonical miss 数量，按 query 顺序从 `free_slots` 分配 local slot。
6. 更新 `index` 和 `slot_to_index`，canonical miss 的 `miss_out=1`。
7. duplicate follower 读取 canonical owner 已安装的 slot，但保持 `miss_out=0`。
8. 融合 maintain 阶段从循环 cursor 开始扫描 10K slot：
   - 跳过本轮 query 使用的 protected slot；
   - 选择与 miss 数量相同的 victim；
   - 清除 victim 的双向映射；
   - 用 victim slot 补回 `free_slots`；
   - 更新循环 cursor；
   - 把 `free_head[0]` 恢复为 0，表示事务闭合。

8K 是持续有效的目标工作集大小，2K 是本轮更新的 free/swap 空间。算子在分配 miss 后同时淘汰相同数量的旧项，使有效工作集回到 8K。当前替换策略是从循环 cursor 开始扫描并保护本轮查询 slot，不是维护完整访问时间戳的严格 LRU。

## 8. 新增和改变的数据结构

### 8.1 核心执行结构

定义位置：`vllm_ascend/attention/dsa_sparse.py`

| 数据结构 | 作用 |
| --- | --- |
| `DSASparseCacheConfig` | 固定容量、block size、max sequence 数和 Top-K 宽度 |
| `RequestIndexManager` | `request_id -> stable request_index` 的 admission/release 管理 |
| `DSASparseCohortKey` | cohort 名和 target/draft role |
| `DSASparseLayerKey` | cohort 与 layer 的组合键 |
| `DSASparseLookupState` | 每 cohort 的 `index/slot_to_index/free_slots/free_head` |
| `DSASparseLookupBatch` | 单次调用的 `req_pool_entries/query_index/lookup_mask` |
| `DSASparseLookupOutput` | 算子输出 `slot_out/miss_out` |
| `DSASparseStepMetadata` | 所有 target cohort 共享的当前 active batch metadata |
| `DSASparseLayerLayout` | 一层 Main payload 的 plane dtype 和 row shape |
| `DSASparseLayerHotCache` | 一层独立的 Hot Main planes |
| `DSASparseLayerBinding` | 一层 Hot Cache 与 I/O context/region/completion |
| `DSASparseCohort` | leader 和 cohort lookup state |
| `DSASparseResolution` | 传给 SFA 的本层 Hot Cache、local indices 和 Hot block table |
| `DSASparseEagerStep` | 一个 cohort 的 step 状态和完成集合 |
| `DSASparseEagerBatchContext` | 单 cohort 的执行上下文 |
| `DSASparseEagerContextRouter` | 按 layer 路由到对应 cohort context |

### 8.2 I/O 接口结构

定义位置：`vllm_ascend/attention/dsa_sparse_io.py`

| 数据结构或接口 | 作用 |
| --- | --- |
| `DSASparseIOCapabilities` | 描述后端是否支持 eager、device plan、稳定地址、P/D publication 等能力 |
| `DSASparseStorageLayout` | 持久 Main region 的 layout |
| `DSASparsePortableBlock` | P/D 间可移植的 logical block 身份 |
| `DSASparseRegionKey` | deployment/instance/role/rank/layer 的 region 身份 |
| `DSASparseIOBackend` | capacity、context、region、publication、bind、release 生命周期接口 |
| `DSASparseIOOperator` | 每层统一数据面入口 |
| `DSASparseIOBackendRegistry` | 初始化期 backend factory registry |

生产 I/O 算子预期同时处理：

- 根据 `query_index`、`block_table` 和请求 region 找到 Main 历史 source；
- 按 `miss_out` 选择需要加载的 token；
- 按 `slot_out` 写入本层 Hot Cache；
- 处理当前 token 的 Main 持久化或 publication；
- 在返回前建立 completion 依赖。

这些能力目前都没有真实实现。

### 8.3 P/D 生命周期结构

定义位置：`vllm_ascend/attention/dsa_sparse_pd.py`

| 数据结构 | 作用 |
| --- | --- |
| `DSASparseTransferCompletion` | request ID 与 generation |
| `DSASparseRequestSnapshot` | 当前 handoff 的可观察快照 |
| `_DSASparseRequestState` | 内部 Main-ready、Indexer-ready、region、admission 状态 |
| `DSASparsePDLifecycle` | 双 ready、generation、admit、preempt、finish、abort 状态机 |

### 8.4 Worker 结构

| 数据结构 | 文件 | 作用 |
| --- | --- | --- |
| `DSASparseEagerCohortDescriptor` | `worker/dsa_sparse_eager.py` | runner 可见的 cohort 路由 |
| `DSASparseEagerCohortLayout` | 同上 | cohort 内有序的逐层 Main layout |
| `DSASparseEagerExecution` | 同上 | metadata attach/detach 和 finish/abort |
| `DSASparseEagerRuntime` | 同上 | request lifecycle、batch context 和 mock runtime |
| `DSASparseExternalMainSpecs` | `worker/dsa_sparse_external_main.py` | D worker 内部保存、但不交给 scheduler 的 Main spec |
| `DSASparseFixedHBMBreakdown` | `worker/dsa_sparse_memory.py` | 固定 HBM 的逐项统计 |

## 9. 配置项、环境变量和 mock 项

### 9.1 `additional_config`

启用入口为：

```json
{
  "dsa_sparse_config": {
    "io_backend": "mock",
    "io_backend_options": {}
  }
}
```

用户可配置字段只有：

| 字段 | 当前要求 |
| --- | --- |
| `io_backend` | 必须是 `"mock"` |
| `io_backend_options` | dict；当前 no-op backend 不消费其中的生产参数 |

以下值不是 `dsa_sparse_config` 用户字段，而是从其他 vLLM 配置派生：

- `kv_role`：来自 `kv_transfer_config.kv_role`；
- `index_topk`：来自 GLM-5 model config，必须为 2048；
- `max_num_seqs`：来自 runner；
- `max_model_len`：必须不超过 128K；
- `block_size`：必须同时整除 8K 和 2K。

未知 `dsa_sparse_config` 字段会直接报错。

### 9.2 支持范围校验

| 维度 | 当前约束 |
| --- | --- |
| device | Ascend A5 |
| model | `model_type="glm_moe_dsa"`，并走 sparse SFA |
| runner | V1 |
| execution | `enforce_eager=True`、`cudagraph_mode=NONE`、xlite graph 关闭 |
| P/D | 必须配置 KV connector，role 为 producer 或 consumer |
| D failure policy | `kv_load_failure_policy="fail"` |
| target tokens | 每请求每 step 恰好 1 token |
| speculative decode | 不支持，`num_speculative_tokens=0` |
| PP/PCP/DCP | 都必须为 1 |
| sequence parallel | 不支持 |
| tensor parallel | 配置校验没有强制为 1 |
| D-side batch state | 仅 `DecodeOnly`，不支持 prefill 或 mixed |

### 9.3 环境变量

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `VLLM_ASCEND_DSA_SPARSE_MOCK_SKIP_MOONCAKE` | 0 | 测试时保留 Mooncake control plane 和地址构造，但跳过实际 `batch_transfer_sync_read` |
| `VLLM_ASCEND_DSA_SPARSE_RUNTIME_PROBE` | 0 | 同步设备后输出机器可读 probe 事件 |

probe 事件包括：

- `runtime_ready`；
- `hot_cache_registered`；
- `lookup_update_done`；
- `hot_cache_sfa_done`。

开启 probe 会执行 `torch.npu.synchronize()`，只适合验证，不用于性能测试。

### 9.4 当前 mock 项

| mock 项 | 位置 | 实际行为 |
| --- | --- | --- |
| `MockDSASparseIOOperator` | `attention/dsa_sparse_io.py` | 校验参数契约，不搬运 live-tail 或 history payload |
| `_MockDSASparseIOResource` | `worker/dsa_sparse_eager.py` | 为每层提供 identity-only context/region/completion |
| `_MockDSASparseRequestRegionBackend` | 同上 | `release_request()` 为 no-op |
| mock P/D admission | `DSASparseEagerRuntime.admit_mock_request()` | 立即模拟 Main-ready 与 Indexer-ready |
| Mooncake payload skip | `mooncake_connector.py` | 地址列表生成后直接返回，不发起数据传输 |
| zero-block Main placeholder | `model_runner_v1.py` | 只维持 attention module ABI，不是 payload mock 数据源 |

即使 P/D probe 能生成 completion，也不能据此判断模型输出正确，因为：

- Mooncake payload 可能被跳过；
- Main history miss 没有真实加载；
- Hot Cache 中未被当前 token prolog 写入的历史行没有可信 payload。

## 10. 相对旧设计已经删除或改变的部分

当前最终代码与分支早期方案相比，已做以下简化：

| 旧概念 | 当前状态 |
| --- | --- |
| `CacheSeatLease(seat, epoch)` | 删除，改为 `RequestIndexManager` 的稳定 `request_index` |
| `row_to_seat` | 删除，使用 `req_pool_entries` 直接选择请求行 |
| seat epoch tensor | 删除，request generation 留在 P/D 生命周期控制面 |
| `DSASparseResidencyState` 和独立 plan cache | 删除，简化为 `DSASparseLookupState` 与当次两个输出 |
| `resolved_hot_indices` | 删除，算子输出名和语义改为 `slot_out` |
| `miss_mask` | 改为 int32 `miss_out` |
| `lru_slots` 输入 | 删除，循环 victim maintain 融合进 SIMT kernel |
| 单独 maintain 算子 | 框架不调用；maintain 已融合进 `dsa_sparse_lookup_update` |
| 独立 newest 准备算子 | 不存在；当前 token Main 写入复用现有 SFA prolog |
| 多 query token/投机 batch | 当前不支持，每请求每 step 固定 1 个 target token |
| 持久 lookup plan/workspace | 不存在；输出按调用分配，workspace 由 CANN tiling 临时申请 |

## 11. 生产代码修改位置及目的

### 11.1 Python 框架与 runtime

| 文件 | 修改目的 |
| --- | --- |
| `vllm_ascend/dsa_sparse_config.py` | 新增 feature config 解析和支持范围校验 |
| `vllm_ascend/dsa_sparse_constants.py` | 定义 128K/8K/2K/10K/2K 和 free-head stride |
| `vllm_ascend/dsa_sparse_probe.py` | 输出同步后的机器可读执行事件 |
| `vllm_ascend/ascend_config.py` | 把 DSA Sparse 配置加载到 `AscendConfig` |
| `vllm_ascend/platform.py` | 在引擎初始化时校验 A5、GLM-5、eager、V1、P/D 和并行限制 |
| `vllm_ascend/envs.py` | 注册 Mooncake mock skip 和 runtime probe 环境变量 |
| `vllm_ascend/core/kv_cache_interface.py` | 把 Main spec 与 Indexer spec 分离；新增 `AscendSFAIndexerCacheSpec` 和对应 manager/backend |
| `vllm_ascend/attention/indexer.py` | 新增 cache-only Indexer backend 和不产生 attention metadata 的 builder |
| `vllm_ascend/ops/mla.py` | `IndexerWrapper` 保留独立 `k_cache`，只删除旧 Top-K buffer |
| `vllm_ascend/attention/sfa_v1.py` | 组合 split Main/Indexer cache；把 Main 写入重定向到 Hot Cache；调用 DSA context、逐层 I/O 和 Hot Cache SFA |
| `vllm_ascend/attention/dsa_sparse.py` | 核心 request index、cohort、lookup state、Hot Cache、step、coordinator 和 context router |
| `vllm_ascend/attention/dsa_sparse_io.py` | 定义生产 I/O backend/operator 契约和 no-op mock |
| `vllm_ascend/attention/dsa_sparse_pd.py` | 定义 P/D 双 ready、generation、admission 和 retire 生命周期 |
| `vllm_ascend/ops/dsa_sparse.py` | Python 到 `_C_ascend.dsa_sparse_lookup_update` 的紧凑适配层 |
| `vllm_ascend/worker/dsa_sparse_eager.py` | 按 `skip_topk` 构建 cohort，分配逐层 Hot Cache，管理 mock runtime 和 metadata attach |
| `vllm_ascend/worker/dsa_sparse_external_main.py` | 从 D scheduler 视图移除 Main tensor，同时维持 worker 内部格式元数据 |
| `vllm_ascend/worker/dsa_sparse_memory.py` | 计算逐层 Hot payload 与逐 cohort lookup state 的固定 HBM |
| `vllm_ascend/worker/model_runner_v1.py` | 主集成点：外部 Main、固定内存、runtime 初始化、请求 admission、forward context |
| `vllm_ascend/worker/worker.py` | 从显式和自动 KV cache memory budget 中扣除 DSA 固定 HBM |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | 增加测试专用的 payload transfer skip |
| `vllm_ascend/utils.py` | 移除已经属于旧混合 Main/Indexer spec 的 helper |

### 11.2 C++、CANN 与构建

| 文件或目录 | 修改目的 |
| --- | --- |
| `csrc/attention/dsa_sparse_lookup_update/CMakeLists.txt` | 定义单算子构建目标 |
| `csrc/attention/dsa_sparse_lookup_update/dsa_sparse_lookup_update_torch_adpt.h` | Torch NPU adapter 和输入契约校验 |
| `csrc/attention/dsa_sparse_lookup_update/op_host/CMakeLists.txt` | host/ACLNN 构建配置 |
| `csrc/attention/dsa_sparse_lookup_update/op_host/dsa_sparse_lookup_update_def.cpp` | CANN op definition |
| `csrc/attention/dsa_sparse_lookup_update/op_host/dsa_sparse_lookup_update_infershape.cpp` | shape 和 dtype inference |
| `csrc/attention/dsa_sparse_lookup_update/op_host/dsa_sparse_lookup_update_tiling.cpp` | 固定 shape 校验、AIV blockDim 和 workspace |
| `csrc/attention/dsa_sparse_lookup_update/op_host/dsa_sparse_lookup_update_tiling.h` | tiling data |
| `csrc/attention/dsa_sparse_lookup_update/op_host/op_api/aclnn_dsa_sparse_lookup_update.cpp`、`csrc/attention/dsa_sparse_lookup_update/op_host/op_api/aclnn_dsa_sparse_lookup_update.h` | ACLNN 导出接口 |
| `csrc/attention/dsa_sparse_lookup_update/op_kernel/dsa_sparse_lookup_update_common.h` | kernel 固定常量和 workspace layout |
| `csrc/attention/dsa_sparse_lookup_update/op_kernel/dsa_sparse_lookup_update.cpp` | AIV-only kernel 入口 |
| `csrc/attention/dsa_sparse_lookup_update/op_kernel/arch35/dsa_sparse_lookup_update_simt.h` | 融合 lookup、分配、duplicate 处理和 maintain |
| `csrc/torch_binding.cpp` | 注册 PrivateUse1 Torch schema |
| `csrc/torch_binding_meta.cpp` | 注册 Meta 实现，供符号 tracing 使用 |
| `csrc/build_aclnn.sh` | 把算子加入 Ascend 950 custom op 构建列表 |

### 11.3 安装、CI、示例和验证资产

| 文件或目录 | 修改目的 |
| --- | --- |
| `setup.py` | `setuptools_scm` 只匹配 `v[0-9]*` release tag，避免 checkpoint tag 破坏 editable install |
| `.github/workflows/scripts/test_config.yaml` | 把 Indexer backend 纳入 SFA 测试依赖和路由 |
| `examples/dsa_sparse_pd_mock_probe.sh` | 同容器双卡 P/D、proxy、请求、runtime probe 和 Decode profiler 的端到端脚本 |
| `examples/dsa_sparse_probe_validate.py` | 校验每 cohort lookup 次数、每层独立 Hot Cache、每层 SFA 事件和 profiler 中的自定义算子 |
| `tools/dsa_sparse_lookup_update/build_and_install.sh` | 单独构建并安装 Ascend 950 算子 |
| `tools/dsa_sparse_lookup_update/common.py` | standalone runtime、输入构造和调用公共逻辑 |
| `tools/dsa_sparse_lookup_update/test_correctness.py` | 对照 CPU oracle 校验输出及四个持久 state tensor |
| `tools/dsa_sparse_lookup_update/benchmark_operator.py` | 固定 8K resident、可配置并发和 miss rate/count 的 event timing |
| `tools/dsa_sparse_lookup_update/profile_operator.py` | steady all-hit workload 和 `torch_npu.profiler` trace |
| `tools/dsa_sparse_lookup_update/README.md` | standalone 工具使用说明和计时边界 |
| `tools/dsa_sparse_lookup_update/.gitignore` | 忽略本地安装和 profile 产物 |

### 11.4 测试文件

| 测试组 | 文件 | 覆盖内容 |
| --- | --- | --- |
| Cache split/SFA | `tests/ut/attention/a2/test_sfa_v1.py`、`tests/ut/attention/test_indexer.py`、`tests/ut/ops/test_mla.py` | Main/Indexer 组合、Hot Cache SFA 和 wrapper 绑定 |
| Core runtime | `tests/ut/attention/test_dsa_sparse.py`、`tests/ut/attention/test_dsa_sparse_eager.py` | request index、cohort、step、逐层资源和调用顺序 |
| I/O/PD | `tests/ut/attention/test_dsa_sparse_io.py`、`tests/ut/attention/test_dsa_sparse_pd.py` | I/O contract、generation 和 dual-ready 生命周期 |
| Config/platform | `tests/ut/test_dsa_sparse_config.py`、`tests/ut/test_platform.py` | 支持矩阵和拒绝条件 |
| Worker/memory | `tests/ut/worker/test_dsa_sparse_eager_runtime.py`、`tests/ut/worker/test_dsa_sparse_memory.py`、`tests/ut/worker/a2/test_model_runner_v1.py` | runtime 构建、固定 HBM、external Main 和 forward 接入 |
| Operator reference | `tests/ut/ops/dsa_sparse_lookup_update_reference.py`、`tests/ut/ops/test_dsa_sparse_lookup_update_reference.py` | CPU oracle 和融合 maintain 语义 |
| Operator source/Torch | `tests/ut/ops/test_dsa_sparse_lookup_update_kernel_source.py`、`tests/ut/ops/test_dsa_sparse_lookup_update_torch.py` | kernel ABI、已删除旧参数、Torch schema |
| Mooncake mock | `tests/ut/kv_offload/test_mooncake_connector.py` | payload skip 开关 |
| Probe validator | `tests/ut/test_dsa_sparse_probe_validate.py` | 事件计数、地址隔离和 profile 识别 |

## 12. 单算子和端到端验证能力

### 12.1 Correctness

```bash
python3 tools/dsa_sparse_lookup_update/test_correctness.py \
  --install-root tools/dsa_sparse_lookup_update/.install \
  --device npu:0 \
  --requests 2 \
  --random-cases 10
```

它对照 CPU oracle 检查：

- `slot_out`；
- `miss_out`；
- `index`；
- `slot_to_index`；
- `free_slots`；
- `free_head`；
- hit、mask、invalid、duplicate miss、reordered `req_pool_entries`；
- fused eviction、free-list refill、cursor 移动和事务 head 归零。

### 12.2 Benchmark

```bash
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --concurrency 8 \
  --miss-rate 10 \
  --warmup 10 \
  --iterations 100
```

或使用：

```bash
--miss-count 205
```

计时区间只包含一次 batched custom-op invocation。输入构造、初始 state、query group 切换、外部同步和 JSON 序列化不在 event timing 内。

### 12.3 Profile

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0 \
  --requests 8
```

profile workload 是固定 2K query 的 steady all-hit lookup。脚本检查解析后的 profile 中是否出现 `DsaSparseLookupUpdate`。

### 12.4 P/D 端到端 probe

P/D 脚本使用 runtime 事件和 Decode profiler 同时验证：

- runtime 建立了多少 cohort 和 layer；
- 每层 Hot Cache 地址互不重叠；
- 每个 completion token 每个 cohort 恰好一次 lookup；
- 每个 completion token 每层恰好一次 Hot Cache SFA；
- profile 中存在自定义算子。

它验证的是调用路径和资源隔离。由于 I/O 与 Mooncake payload 可以被 mock，不能验证 Main history payload 或模型精度。

## 13. 当前仍未完成的工作

### 13.1 生产 I/O

必须实现真实的 `DSASparseIOBackend` 和 `DSASparseIOOperator`：

- P 侧 Main region publication；
- D 侧 portable block 到本地 region 的 bind；
- 根据 `query_index` 和 block table 生成真实 source address；
- 按 `miss_out` 批量读取；
- 按 `slot_out` 写入逐层 Hot Cache；
- 当前 token Main 的持久化；
- stream/event/completion 依赖；
- 请求 finish/preempt 的安全回收。

这是当前端到端数值正确性的最大缺口。

### 13.2 真实 P/D ready 接入

需要把：

- Main publication completion；
- Indexer KV transfer completion；
- scheduler admission；
- request generation；

连接到 `DSASparsePDLifecycle`。当前 runner 使用的即时 mock admission 不能进入生产路径。

### 13.3 图模式

当前配置层明确拒绝 CUDAGraph 和 xlite graph。要支持 graph，需要解决：

- 固定地址输入输出的 graph binding；
- lookup state 原地更新的捕获语义；
- I/O 外部异步执行与 graph completion；
- dynamic active request 数；
- step 生命周期与 graph replay；
- probe/debug 路径与 graph 路径隔离。

虽然 Torch Meta 实现已经存在，但它只解决 schema/tracing 入口，不代表完整 graph 可运行。

### 13.4 多 token 与投机推理

当前每请求每 step 只接受一个 target token。要支持 speculative decode，需要单独定义：

- 多 query token 的 semantic Top-K；
- draft 与 target 的 Hot Cache ownership；
- 多 token live-tail 排布；
- accepted/rejected token 回滚；
- lookup state 和 Main payload 的提交边界。

### 13.5 并行模式

PP、PCP、DCP 和 sequence-parallel 当前都被拒绝。未来支持时必须明确 request index、cohort state、Main region、Top-K 和 Hot slot mapping 在各 rank 之间的分片或复制规则。

## 14. 提交演进

33 个提交可以分为六个阶段：

| 阶段 | 主要提交 | 结果 |
| --- | --- | --- |
| Cache split | `a99b89abdb` | Main 与 Indexer KV cache spec、backend 和 binding 解耦 |
| eager 框架 | `4b6ebc0d00` 至 `aac21b73d5` | config、P/D lifecycle、Hot Cache、batch context、external Main 和固定 tensor |
| Graph-out 闭环与构建修复 | `4ac6006f1e` 至 `5f685a1a14` | 接入 eager forward、修复安装/tiling、加入 P/D mock 和 SFA rank 适配 |
| 路径验证与 standalone 工具 | `66d8a7b7e7` 至 `159aebd455` | probe、correctness、benchmark、profile 和 miss 控制 |
| ASU 接口简化 | `fa5e02d7de` 至 `67250d1235` | 稳定 request index、ASU lookup ABI、融合 maintain、测试重构 |
| SIMT workspace 修复 | `74f00dddc7` | tiling 同时预留 CANN system workspace 与 per-request user workspace |

完整提交顺序如下：

```text
a99b89abdb refactor(attention): split SFA indexer KV cache
4b6ebc0d00 feat(attention): add DSA sparse eager cache state
c9b095812d feat(attention): add DSA sparse eager I/O flow
e24f1aba9f feat(config): gate DSA sparse eager P/D mode
ac089495c0 feat(attention): add DSA sparse P/D ready lifecycle
ac1440e12c feat(attention): add DSA sparse eager batch contexts
1647d61b4b feat(attention): route DSA sparse eager through Hot Cache
83fbf7bf94 feat(worker): add DSA sparse eager batch runtime
55eb340169 feat(worker): enter DSA sparse eager runtime
ce8c790222 fix(attention): constrain DSA sparse target eager flow
923e2ae8ea feat(worker): externalize DSA sparse Decode Main cache
aac21b73d5 feat(worker): allocate DSA sparse eager fixed tensors
4ac6006f1e feat: complete DSA sparse eager graph-out flow
2014a3544d fix(build): ignore non-version tags in SCM versioning
4d2375e232 fix(dsa-sparse): use supported tiling log macro
cf03c4ce28 fix(dsa-sparse): preserve mutable platform info type
06c8ea7d46 test(dsa-sparse): add same-node PD mock probe
b2c472036a test(dsa-sparse): allow mocking Mooncake payload transfer
3434464e73 fix(dsa-sparse): register Mooncake mock environment flag
d03d70386c fix(dsa-sparse): align staged query position dtype
80ca262d82 fix(dsa-sparse): use default kernel tiling entry
5f685a1a14 fix(dsa-sparse): restore SFA sparse index rank
66d8a7b7e7 test(dsa-sparse): verify custom op and hot cache path
62d7122e5c test(dsa-sparse): add standalone custom op tools
80d9535257 fix(dsa-sparse): reset stale single-op CMake cache
0940250694 test(dsa-sparse): add 8k cache operator benchmark
159aebd455 test(dsa-sparse): control benchmark miss rate
fa5e02d7de refactor(dsa-sparse): use stable request indices
57a9e6bb25 fix(dsa-sparse): initialize inactive query outputs
97e26f4afe refactor(dsa-sparse): adapt framework to ASU lookup
479bb8f894 refactor(dsa-sparse): fuse ASU lookup and maintain
67250d1235 test(dsa-sparse): adapt validation to fused lookup
74f00dddc7 fix(dsa-sparse): reserve system workspace for SIMT lookup
```

## 15. 最终结论

相对 `dsa-sparse-0.23`，当前分支已经完成：

- Main/Indexer cache 拆分；
- D scheduler 只管理 Indexer、worker 管理逐层 Main Hot Cache；
- 稳定 `request_index` 和直接 `req_pool_entries` 寻址；
- 按 `skip_topk` 建立 IndexCache cohort；
- Ascend 950 SIMT 融合 lookup/maintain 算子；
- 每 cohort lookup、每层 I/O、每层 Hot Cache SFA 的 eager 调用拓扑；
- 固定 HBM 预算；
- mock P/D 生命周期；
- 单算子 correctness/benchmark/profile 与端到端 probe。

当前没有完成：

- 生产 Main payload I/O；
- 真实 P/D Main publication 和 dual-ready admission；
- graph；
- speculative/multi-token；
- PP/PCP/DCP/SP；
- mock 关闭后的完整数值正确性闭环。

因此当前分支的准确定位是：

> 一个已经接入真实融合 metadata 算子、具备逐层 Hot Main Cache 和完整 eager 调用拓扑，但 Main payload 数据面仍由 mock 占位的 DSA Sparse Graph-out 框架实现。
