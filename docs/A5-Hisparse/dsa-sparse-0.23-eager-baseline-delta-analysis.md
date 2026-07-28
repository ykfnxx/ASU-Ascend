# dsa-sparse-0.23-eager 相对基线修改分析

## 1. 分析范围

- 仓库：`vllm-ascend`
- 基线分支：`dsa-sparse-0.23`
- 基线提交：`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`
- 当前分支：`dsa-sparse-0.23-eager`
- 当前提交：`66d8a7b7e7854621f0955f8ca122fa32d22381fb`
- Merge base：`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`
- 修改规模：23 个线性提交、56 个文件，约 `+12334/-574`

当前分支实现的是一套仅支持 eager Decode 的 DSA Sparse 框架路径：

1. 将 SFA 的 Indexer KV 与 Main MLA KV 拆分成两类 cache。
2. D 节点的 Main KV 不再由 scheduler KV block pool 分配，改为 worker 固定分配的每层 Hot Cache。
3. 新增稳定 cache seat、residency 映射、近似 LRU 和固定执行 plan。
4. 接入真实 Ascend 950 SIMT 融合算子 `dsa_sparse_lookup_update`。
5. 每个 IndexCache cohort 的 leader 每个 step 调用一次 lookup/update；followers
   只复用 semantic Top-K、token-to-slot 驻留映射和本轮 lookup plan，不共享
   Main KV payload。
6. 每个 sparse layer 都独立持有 Hot Main Cache 和 I/O 资源，独立调用一次统一
   I/O 接口，然后基于本层 Hot Cache 调用 SFA。
7. 当前 I/O 仍为 mock；P/D Main 数据发布和历史 Main miss 读取没有真实实现。
8. 增加 Mooncake 跳过开关和运行时探针，用于验证自定义算子及 Hot Cache 路径确实执行。

## 2. 整体架构

### 2.1 初始化阶段

1. 解析 `additional_config.dsa_sparse_config`。
2. 校验 A5、GLM-MOE-DSA、eager、V1 runner、P/D role 等约束。
3. 将 SFA cache 拆成：
   - Indexer cache：仍由 scheduler 分配和管理。
   - Main cache：D 节点从 scheduler 视图中移除。
4. 根据模型层顺序和 `skip_topk` 划分 IndexCache cohort：
   - `skip_topk=False` 的层建立新 cohort，并成为 leader；
   - 后续连续的 `skip_topk=True` 层加入最近的 cohort，成为 followers；
   - 第一层不能是 follower。
5. 预先计算并从 KV block pool 可用内存中扣除固定 HBM。
6. 分配：
   - 所有 target cohort 共享的 batch metadata；
   - 每个 cohort 的 residency state、lookup plan 和 workspace；
   - 每层独立的 Hot Main Cache；
   - mock I/O context、region 和 completion。
7. 给 externalized Main layer 绑定零 block placeholder，只维持模型模块的 KV cache ABI。

### 2.2 请求生命周期

当前真正接入 model runner 的是 mock 生命周期。

新请求或恢复请求：

1. 模拟 Main region ready。
2. 模拟 Indexer ready。
3. 分配稳定的 `CacheSeatLease(seat, epoch)`。

完成、抢占或再次恢复：

1. 检查该请求没有活跃 step。
2. 释放 mock region。
3. 释放 cache seat。

通用的双 ready 生命周期已经定义，但没有与真实 Mooncake/Main 后端完成事件接通。

### 2.3 每个 Decode step

```text
scheduler metadata
    |
    v
request row -> stable cache seat
    |
    v
生成 Main block_table、Hot block_table、newest write descriptor
    |
    v
SFA prolog 将当前 token Main KV 写入本层 reserved Hot slot
    |
    v
cohort leader 产生 semantic Top-K
    |
    v
dsa_sparse_lookup_update（每个 cohort 一次）
    |-- 更新 token_to_hot / hot_to_token / LRU
    |-- 生成 resolved_hot_indices
    `-- 生成 miss_mask
    |
    |  同一 cohort 的 followers 复用上述 semantic Top-K、映射和 plan
    |  但各层不共享 Main KV payload
    |
    v
dsa_sparse_io（每层一次，写入本层 Hot Main Cache，当前为 mock）
    |
    v
SparseFlashAttention（每层一次）
    |-- 本层 Hot Main Cache
    |-- [Q, 1, K] local sparse indices
    `-- synthetic Hot block table
```

这里需要区分：

- Main KV payload 是逐层独立的。每层有自己的 Hot Cache planes、I/O context、
  backend region 和 completion，并分别执行 I/O 与 SFA。
- `token_to_hot`、`hot_to_token`、LRU、`state_seat_epoch` 以及本轮 lookup plan
  不是逐层分配，而是由 IndexCache cohort 共享。
- cohort 内共享的是 token position 到 local hot slot 的编号关系。例如
  `token_position=100 -> hot_slot=7` 对 cohort 内各层相同，但 `layer 0` 的
  slot 7 保存 layer 0 的 Main KV，`layer 1` 的 slot 7 保存 layer 1 的
  Main KV；二者是不同的 tensor 地址和不同的 payload。
- 如果多个连续层通过 `skip_topk=True` 复用 leader 的 semantic Top-K，则它们
  也复用 leader 的 lookup 结果；如果每层都是 leader，lookup 才表现为逐层调用。

### 2.4 “逐层 Cache”与 IndexCache cohort 的准确边界

`IndexCache cohort` 不是 scheduler 的 KV cache group、block table 类型或共享
payload 区域，而是当前分支在 DSA Sparse eager runtime 中新增的执行分组。它表示：

```text
一个产生 semantic Top-K 的 leader layer
    +
连续复用该 Top-K 的 skip_topk follower layers
```

当前资源所有权如下：

| 资源 | 当前所有权与行为 |
| --- | --- |
| Main/MLA Hot Cache planes | 每层独立分配 |
| I/O context、region、completion | 每层独立 |
| 每层 Main KV 的 newest write | 写入本层 Hot Cache 的 reserved slot |
| 每层 I/O 与 SFA 调用 | 每层每 step 各调用一次 |
| scheduler 中的 Indexer KV cache | 按实际 Indexer cache layer 注册和管理，不由 `DSASparseCohort` 分配 |
| semantic Top-K | leader 产生，`skip_topk` followers 复用 |
| `token_to_hot/hot_to_token/LRU/state_seat_epoch` | 每个 cohort 一份 |
| `resolved_hot_indices/miss_mask/workspace` | 每个 cohort、每种 plan key 一份 |
| `cache_seat` | request 生命周期内稳定；同一个 seat 用来寻址各层各自的 Hot Cache 行 |

因此，当前分支应描述为：

> Main KV 数据空间逐层独立，但 cohort 内各层的 Hot Cache slot 布局保持同步，
> 并共享 token-to-slot 驻留状态、淘汰顺序和 lookup plan。

共享驻留映射依赖以下不变量：

1. cohort 内所有 follower 使用与 leader 相同的 semantic Top-K token positions。
2. 同一个 token position 在所有层中使用相同的 local hot slot 编号。
3. 每层 I/O 都必须依据同一份 `resolved_hot_indices/miss_mask`，把该层自己的
   Main KV 填入该层 Hot Cache 的对应 slot。
4. 在本轮 SFA 开始前，每层自己的 I/O completion 必须满足；一层完成不能代替
   另一层完成。
5. cohort 内不能出现某一层独立淘汰、独立改变 slot 映射或使用不同 Top-K 的行为。

当前 mock I/O 不搬运 history miss payload，因此只能验证调用拓扑和地址隔离，
不能验证上述 payload 同步不变量在真实 I/O 下成立。

如果设计中的“逐层 Cache”只要求每层 KV payload、region 和 completion 独立，
当前实现符合该要求。如果它还要求每层独立维护 resident mapping、LRU、miss 和
淘汰决策，则当前实现不符合这一更强定义：代码把“共享 semantic Top-K”进一步
扩展成了“共享 physical slot mapping”。严格逐层驻留的实现应只保留 Top-K
结果共享，并为每层分别分配 `DSASparseResidencyState`、`DSASparsePlan`，分别
调用 `dsa_sparse_lookup_update`。

## 3. 主要修改位置及目的

| 位置 | 修改目的 |
| --- | --- |
| `vllm_ascend/core/kv_cache_interface.py` | 将原来混合描述 Main 和 Indexer 的 `AscendMLAAttentionSpec` 改为 Main-only；新增 `AscendSFAIndexerCacheSpec`，让 scheduler 能独立管理 Indexer cache。 |
| `vllm_ascend/attention/indexer.py` | 新增 cache-only Indexer backend；提供 cache shape 和 metadata builder，本身不执行 attention forward。 |
| `vllm_ascend/ops/mla.py` | `IndexerWrapper` 保留 `indexer.k_cache`，不再将其删除，使拆分后的 Indexer cache 能单独绑定。 |
| `vllm_ascend/attention/sfa_v1.py` | 给 metadata 增加 `dsa_sparse_context`；组合 Main 与 Indexer cache；将 Main 写入重定向到 Hot Cache；执行 lookup/I/O 后使用 Hot Cache 调 SFA。 |
| `vllm_ascend/attention/dsa_sparse.py` | 实现核心数据面：cache seat、residency、plan、Hot Cache、cohort、step、coordinator、batch context 和 router。 |
| `vllm_ascend/attention/dsa_sparse_io.py` | 定义真实 I/O 后端应满足的控制面和统一数据面接口；提供当前 no-op mock。 |
| `vllm_ascend/attention/dsa_sparse_pd.py` | 定义 Main-ready 与 Indexer-ready 汇合、generation 校验、seat admission、finish/preempt 生命周期。 |
| `vllm_ascend/dsa_sparse_config.py` | 解析配置并限制当前支持范围：GLM-MOE-DSA、eager、P/D、无 speculative、`io_backend=mock`。 |
| `vllm_ascend/ascend_config.py` | 将 DSA Sparse 配置加载到 `AscendConfig`。 |
| `vllm_ascend/platform.py` | 在引擎初始化阶段拒绝不支持的设备、graph、V2 runner、SP/CP 和非 P/D 配置。 |
| `vllm_ascend/worker/dsa_sparse_external_main.py` | 保存 D 节点 worker-local Main spec；只向 worker 的 KV group 副本补充格式元数据，不新增 scheduler cache tensor。 |
| `vllm_ascend/worker/dsa_sparse_memory.py` | 计算 Hot payload、residency、plan、workspace、初始化 scratch 和执行峰值所需固定 HBM。 |
| `vllm_ascend/worker/dsa_sparse_eager.py` | 构建 cohort/runtime，绑定真实 lookup 算子和 mock I/O，并管理 metadata context 的 attach、finish、abort。 |
| `vllm_ascend/worker/model_runner_v1.py` | 主集成点：识别 cohort、生成层布局、计算 HBM、externalize Main、初始化 runtime、管理 mock 请求，并在 forward 外层进入 eager execution。 |
| `vllm_ascend/worker/worker.py` | 在自动内存 profiling 和显式 `kv_cache_memory_bytes` 两条路径中扣除 DSA Sparse 固定 HBM。 |
| `vllm_ascend/ops/dsa_sparse.py` | Python 到 `torch.ops._C_ascend.dsa_sparse_lookup_update` 的适配层，同时发送 lookup 完成探针。 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | 增加测试用开关，在构造出传输地址后跳过 `batch_transfer_sync_read`。 |
| `vllm_ascend/dsa_sparse_probe.py` | 输出同步后的机器可读事件，用于证明算子和 Hot Cache SFA 路径确实完成。 |
| `setup.py` | `setuptools_scm` 只匹配 `v[0-9]*` tag，避免 checkpoint tag 导致 editable install 失败。 |

### 3.1 Indexer 与 Main cache 拆分

#### `vllm_ascend/core/kv_cache_interface.py`

`AscendMLAAttentionSpec` 不再包含：

- `sparse_head_dim`
- `cache_sparse_li_c8`
- `sfa_dcp_replicated_indexer_size`
- Indexer K/scale 对应的 page size 计算

新增 `AscendSFAIndexerCacheSpec`，独立描述：

- Indexer K dtype 和 head size；
- Indexer scale；
- LI C8；
- DCP replicated indexer 数量；
- Indexer page size；
- 对应的 `FullAttentionManager`。

#### `vllm_ascend/attention/indexer.py`

新增 `AscendSFAIndexerBackend`：

- cache shape 为 `[num_blocks, block_size, num_kv_heads, head_size]`；
- 支持 block size 128；
- metadata builder 不生成 attention metadata；
- 该 backend 只为独立 Indexer cache 提供 scheduler/runner 注册，不执行 attention。

#### `vllm_ascend/worker/model_runner_v1.py`

`get_kv_cache_spec()` 在 sparse SFA 下分别生成：

- Main layer 的 `AscendMLAAttentionSpec`；
- `model.layers.N.self_attn.indexer.k_cache` 的 `AscendSFAIndexerCacheSpec`。

在 D consumer 上：

- Main spec 写入 `DSASparseExternalMainSpecs`；
- 返回给 scheduler 的 `kv_cache_spec` 中只保留 Indexer spec；
- worker 初始化时把 Main spec 补到 worker-local group metadata；
- `runner_only_attn_layers` 使 Main layer 跳过 scheduler cache tensor 分配和 reshape；
- Main module 最终绑定零 block placeholder。

### 3.2 D 节点两类 block table

`DSASparseBatchMetadata` 同时保存两类 block table：

1. `block_table`
   - 来自 Decode scheduler metadata。
   - 表示请求的逻辑历史 token 对应的 Decode physical block。
   - 用于生成 `write_global_slots`，未来真实 I/O 也使用它定位 Main 历史 payload。

2. `hot_block_table`
   - worker 根据稳定 cache seat 合成。
   - 每个 seat 对应固定、连续的一段 Hot blocks。
   - 传给 SparseFlashAttention，用于解释 `resolved_hot_indices`。

两者的行都对应当前 step 的 request row，但底层存储和含义不同。

### 3.3 固定 HBM 管理

`DSASparseFixedHBMBreakdown` 统计：

- `hot_payload_bytes`
- `batch_metadata_bytes`
- 每 cohort 的 `residency_state_bytes`
- 每 cohort 的 `lookup_plan_bytes`
- `initialization_scratch_bytes`
- `eager_batch_staging_bytes`
- 每 cohort 的 eager context 和 lookup scratch
- `backend_auxiliary_bytes`

最终：

```text
fixed_hbm
  = core_fixed_tensor_bytes
  + max(initialization_peak, eager_execution_peak)
  + backend_auxiliary_bytes
```

worker 在 KV block 数量确定前，从以下两条路径的可用预算中扣除固定 HBM：

- 自动 memory profiling；
- 用户显式设置的 `kv_cache_memory_bytes`。

当前 mock backend 的 `backend_auxiliary_bytes=0`。

## 4. 自定义算子

算子目录：

```text
csrc/attention/dsa_sparse_lookup_update/
```

文件作用：

| 文件 | 作用 |
| --- | --- |
| `op_host/dsa_sparse_lookup_update_def.cpp` | 注册 16 个输入和 Ascend 950 target。 |
| `op_host/dsa_sparse_lookup_update_tiling.cpp` | 校验 tensor shape、`T*K <= S`、workspace，并设置 AIV block 数。 |
| `op_host/dsa_sparse_lookup_update_tiling.h` | 定义 8 个 tiling 维度。 |
| `op_kernel/dsa_sparse_lookup_update.cpp` | 按 request row 分配 AIV，启动 256-thread SIMT。 |
| `op_kernel/arch35/dsa_sparse_lookup_update_simt.h` | 实际查找、去重、victim 选择、映射维护和近似 LRU。 |
| `op_kernel/dsa_sparse_lookup_update_common.h` | Host/kernel 共享常量。 |
| `op_host/op_api/*` | ACLNN executor 接口。 |
| `dsa_sparse_lookup_update_torch_adpt.h` | Torch C++ 调用适配。 |
| `csrc/torch_binding.cpp` | 注册带 mutation annotation 的 Torch schema。 |
| `csrc/torch_binding_meta.cpp` | Meta backend 空实现。 |
| `csrc/build_aclnn.sh` | 将算子加入自定义算子编译列表。 |

算子原地修改：

- `token_to_hot`
- `hot_to_token`
- `lru_slots`
- `state_seat_epoch`
- `resolved_hot_indices`
- `miss_mask`
- `workspace`

主要语义：

1. 按 `query_to_row/query_to_lane` 找到当前 request row 的 query。
2. 根据 seat epoch 延迟清空被重新分配的 seat。
3. 当前 step 的 query position 映射到 reserved newest slot。
4. 查找已有 resident token。
5. 对同一 Top-K token 的重复 miss 做确定性去重。
6. 保护当前 Top-K union 中的已有 resident slot。
7. victim 选择采用空闲 slot 优先，再按旧 LRU 顺序选择占用 slot。
8. 更新双向映射和 approximate-LRU。
9. 让重复项复用 canonical miss 安装出的 slot。

该算子只管理索引和 residency，不搬运 Main payload。

## 5. 新增数据结构

### 5.1 Cache 与寻址

#### `DSASparseCacheConfig`

保存：

- 最大请求数；
- 最大模型长度；
- block size；
- evictable Hot slot 数；
- query lane 数；
- Top-K。

推导：

- `reserved_newest_slots`
- `alignment_padding_slots`
- `managed_hot_width`
- `hot_stride`
- `hot_blocks_per_seat`
- `total_hot_blocks`
- `max_blocks_per_request`

`device_buffer_size` 只包含可淘汰 Hot slot，不包含 reserved newest slot 和 alignment padding。

#### `CacheSeatLease`

字段：

- `seat`
- `epoch`

epoch 用于避免 seat 复用后读取上一任请求的 residency 状态。

#### `DSASparseRowMapping`

字段：

- `row_to_cache_seat`
- `row_seat_epoch`

#### `CacheSeatManager`

在控制面维护：

```text
request_id -> CacheSeatLease(seat, epoch)
```

request row 可以随 scheduler batch 重排，但请求持有的 seat 在请求生命周期内保持稳定。

#### `DSASparseResidencyState`

每个 cohort 独立保存：

- `token_to_hot[N, max_model_len]`
- `hot_to_token[N, S]`
- `lru_slots[N, S]`
- `state_seat_epoch[N]`

这里的“每个 cohort”不能写成“每个 layer”。同一 cohort 的 leader/followers
共享这四个驻留张量，因此它们对同一 token position 使用相同的 local hot slot
编号和相同的淘汰顺序；各层对应 slot 中存放的 Main KV payload 仍然独立。

### 5.2 固定执行计划

#### `DSASparsePlanKey`

固定：

```text
token_capacity = request_capacity * query_lane_capacity
```

同时区分 target/draft role。

#### `DSASparseBatchMetadata`

所有 target cohort 共享：

- row mapping；
- query position；
- query 到 row/lane 的映射；
- query valid mask；
- sequence length；
- Main block table；
- synthetic Hot block table；
- newest source/destination descriptor。

#### `DSASparsePlan`

每个 cohort 独立：

- `valid_topk_counts`
- `topk_positions`
- `resolved_hot_indices`
- `miss_mask`
- SIMT workspace

### 5.3 Cohort 与每层资源

- `DSASparseCohortKey`：定义 cohort 名称和 target/draft role。
- `DSASparseLayerKey`：唯一标识 cohort 中的一层。
- `DSASparseLayerLayout`：描述每层 Main cache plane 的 dtype 和每 token shape。
- `DSASparseLayerHotCache`：每层独立持有 Hot Cache planes。
- `DSASparseLayerBinding`：将层绑定到 cohort、Hot Cache、I/O context、region 和 completion。
- `DSASparseCohort`：保存 leader layer、共享 residency state 和 plans。
- `DSASparseResolution`：向 SFA 暴露本层 Hot Main Cache、local sparse indices、synthetic Hot block table。
- `DSASparseEagerStep`：跟踪 lookup、newest write、I/O、SFA 是否完成。
- `DSASparseMainWriteTarget`：将 SFA prolog 写目标切到本层 Hot Cache reserved slot。

cohort 的构造不是按“若干层共享同一块 Main Cache”进行，而是按 IndexCache
Top-K 生产/复用关系进行：

```text
skip_topk=False -> 新 cohort 的 leader
skip_topk=True  -> 最近一个 cohort 的 follower
```

运行时先为 cohort 分配一份 `DSASparseResidencyState` 和 `DSASparsePlan`，
再遍历 cohort 中的所有层，为每层分别分配 `DSASparseLayerHotCache` 和独立的
I/O resources。这形成“cohort 共享索引驻留状态、layer 独占 payload”的两级
所有权。

### 5.4 Runner/runtime

- `DSASparseEagerCohortLayout`
- `DSASparseEagerCohortDescriptor`
- `DSASparseEagerRuntime`
- `DSASparseEagerExecution`
- `DSASparseEagerBatchContext`
- `DSASparseEagerContextRouter`

它们负责从模型层顺序构建 cohort，并将同一个 router 挂到每层：

```text
AscendSFAMetadata.dsa_sparse_context
```

### 5.5 I/O 和 P/D

I/O 合同：

- `DSASparseIOCapabilities`
- `DSASparseStorageLayout`
- `DSASparsePortableBlock`
- `DSASparseRegionKey`
- `DSASparseIOBackend` protocol
- `DSASparseIOOperator` protocol
- `DSASparseIOBackendRegistry`

这些是生产 I/O 的接口定义，目前没有具体 backend。

P/D 生命周期：

- `DSASparseTransferCompletion(request_id, generation)`
- `DSASparseRequestSnapshot`
- `_DSASparseRequestState`
- `DSASparsePDLifecycle`

### 5.6 其他

- `AscendSFAIndexerCacheSpec`：独立 Indexer KV cache spec。
- `DSASparseExternalMainSpecs`：D worker-local Main 格式信息。
- `DSASparseFixedHBMBreakdown`：固定 HBM 各组成部分。
- `DsaSparseLookupUpdateTilingData`：算子侧固定 tiling ABI。

## 6. 新增配置项

### 6.1 `additional_config.dsa_sparse_config`

配置存在即启用。

```json
{
  "dsa_sparse_config": {
    "io_backend": "mock",
    "io_backend_options": {},
    "device_buffer_size": 4096
  }
}
```

只允许三个字段：

| 配置 | 作用 | 当前限制 |
| --- | --- | --- |
| `io_backend` | 选择 Main I/O backend | 只能是 `"mock"` |
| `io_backend_options` | 预留给 backend 的参数 | 当前 runtime 没有使用这些参数 |
| `device_buffer_size` | 每个请求的 evictable Hot slot 数 | 仅 D 节点必填，且至少为 `max_query_tokens_per_request * index_topk` |

从其他配置派生但不属于该字典的字段：

- `kv_role`：来自 `kv_transfer_config.kv_role`。
- `index_topk`：来自模型 HF config。
- `max_query_tokens_per_request`：当前固定为 1。

### 6.2 新增环境变量

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `VLLM_ASCEND_DSA_SPARSE_MOCK_SKIP_MOONCAKE` | `0` | 跳过 Mooncake payload transfer。 |
| `VLLM_ASCEND_DSA_SPARSE_RUNTIME_PROBE` | `0` | 输出 runtime、lookup 和 Hot Cache SFA 探针。 |

注意：Mooncake skip 判断位于通用传输函数中。进程设置为 `1` 后，所有进入该代码段且 `src_list` 非空的 Mooncake read 都会被跳过，并没有再次判断请求是否属于 DSA Sparse。

### 6.3 启用约束

当前配置强制要求：

- Ascend A5；
- `model_type == "glm_moe_dsa"`；
- V1 model runner；
- `enforce_eager=True`；
- `cudagraph_mode=NONE`；
- xlite graph 关闭；
- P/D KV connector 已配置；
- D 节点 `kv_load_failure_policy="fail"`；
- PP、DCP、PCP 都为 1；
- 不支持 speculative tokens；
- 不支持 SP padding；
- 不支持 context-parallel SFA。

TP 没有被该配置直接禁止。

## 7. Mock 项清单

### 7.1 Main I/O mock

`MockDSASparseIOOperator`：

- 每层都会调用；
- 校验 shape、dtype、plane 数和 block table；
- 不读取历史 Main payload；
- 不执行 miss fill；
- 不建立真实异步 completion dependency。

因此 lookup 算子将 miss 安装到某个 Hot slot 后，该 slot 的历史 payload 仍然无效。

### 7.2 Mock I/O 资源

`vllm_ascend/worker/dsa_sparse_eager.py` 中：

- `_MockDSASparseIOResource`
  - 仅用于保证每层 context、region、completion 具有独立 identity。
- `_MockDSASparseRequestRegionBackend`
  - `release_request()` 是 no-op。

### 7.3 Mock P/D 生命周期

`admit_mock_request()` 同步模拟：

1. `begin_handoff`
2. Main region ready
3. Indexer ready
4. ready notification
5. stable seat admission

这只能证明生命周期状态机可以运行，不代表真实 Mooncake completion 已接入。

### 7.4 Mooncake payload skip

设置：

```bash
export VLLM_ASCEND_DSA_SPARSE_MOCK_SKIP_MOONCAKE=1
```

Mooncake 仍执行请求匹配和地址列表构建，但不调用实际的：

```text
batch_transfer_sync_read
```

### 7.5 External Main placeholder

D 节点给 external Main layer 绑定 shape 第一维为 0 的 cache tensor。

它不是实际 Main storage，也不包含 payload，只用于满足模型模块和 forward context 对 `kv_cache` 属性的结构要求。

### 7.6 显式 fail-fast stub

仍保留：

- `UnimplementedDSASparseLookupUpdateOperator`
- `UnimplementedDSASparseIOOperator`

默认 eager mock runtime 使用的 lookup 已经是真实 Torch/Ascend 自定义算子；I/O 默认仍使用 `MockDSASparseIOOperator`。

### 7.7 Mock probe 脚本

`examples/dsa_sparse_pd_mock_probe.sh`：

- 同一节点启动 1P1D；
- 可分别指定两张 NPU；
- 开启 Mooncake skip；
- 开启 runtime probe；
- 发起多 token completion；
- 收集 profiler；
- 调用 `examples/dsa_sparse_probe_validate.py` 校验：
  - 每层 Hot Cache 地址独立；
  - lookup 次数等于 `cohort_count * completion_tokens`；
  - Hot Cache SFA 次数等于 `layer_count * completion_tokens`；
  - SFA 使用的指针与注册 Hot Cache 一致；
  - sparse indices 为 `[Q, 1, K]`；
  - profiler 中存在自定义算子。

该 probe 能证明真实 lookup 算子和 Hot Cache SFA 路径被执行，但不能证明：

- Main payload 传输；
- history miss fill；
- 多步数值正确性；
- 模型精度。

## 8. 测试修改

### 8.1 Cache state、cohort 和 context

- `tests/ut/attention/test_dsa_sparse.py`
- `tests/ut/attention/test_dsa_sparse_eager.py`
- `tests/ut/attention/test_dsa_sparse_pd.py`
- `tests/ut/attention/test_dsa_sparse_io.py`

### 8.2 Runner、生命周期和固定 HBM

- `tests/ut/worker/test_dsa_sparse_eager_runtime.py`
- `tests/ut/worker/test_dsa_sparse_memory.py`
- `tests/ut/worker/a2/test_model_runner_v1.py`

### 8.3 算子语义与 Torch 注册

- `tests/ut/ops/dsa_sparse_lookup_update_reference.py`
- `tests/ut/ops/test_dsa_sparse_lookup_update_reference.py`
- `tests/ut/ops/test_dsa_sparse_lookup_update_torch.py`

### 8.4 Indexer/Main 拆分

- `tests/ut/attention/test_indexer.py`
- `tests/ut/attention/a2/test_sfa_v1.py`
- `tests/ut/ops/test_mla.py`

### 8.5 配置、平台和 mock

- `tests/ut/test_dsa_sparse_config.py`
- `tests/ut/test_platform.py`
- `tests/ut/kv_offload/test_mooncake_connector.py`
- `tests/ut/test_dsa_sparse_probe_validate.py`

## 9. 当前实现边界

当前已经完成：

- eager Decode 框架接线；
- 每层独立 Hot Main Cache、I/O region 和 completion；
- 基于 `skip_topk` 的 cohort Top-K/驻留映射/lookup plan 共享；
- 固定 HBM 预算；
- request seat 生命周期；
- cohort lookup plan；
- 真实 `dsa_sparse_lookup_update`；
- 每层 Hot Cache SFA；
- 可验证的运行时 probe。

仍然是 mock 或未接通的部分：

- 真实 `DSASparseIOBackend`；
- P 节点 Main region 的正式 publication；
- portable block 到 D block table 的正式 bind；
- history miss 的 Main KV 读取；
- I/O completion 和计算流依赖；
- 真实 connector completion 到 `DSASparsePDLifecycle` 的接线；
- graph/capture 路径；
- mock I/O 下的模型数值正确性。

另外，当前实现是否满足最终设计中的“逐层 Cache”，取决于该术语是否包含驻留
索引的所有权：

- 按 payload 所有权定义：满足，每层的 Hot Main KV 和 I/O 资源均独立。
- 按 payload 加 resident mapping/LRU 所有权定义：不满足，mapping、LRU 和
  lookup plan 当前按 IndexCache cohort 共享。
