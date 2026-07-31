# DSA Sparse Offload 迁移方案（v0.19.1rc1 -> G2.5 v0.18 基线）

## 1. 目标与范围

### 1.1 迁移目标

将 `vllm019-DSA-offload` 当前分支 `tmp-opt` 的 DSA Sparse Offload 框架迁移到
`vllm-ascend` 当前分支 `dev_framework_adapt`，保持以下核心功能不变：

1. Indexer KV 完整驻留 HBM，lightning indexer 每层扫描完整历史上下文。
2. MLA KV 只在 HBM 保留固定 sparse budget 和当前未满 tail block。
3. 已完成的 MLA full block 按层卸载到 worker-local DRAM hot store。
4. 每层使用 lightning indexer 的 topK 结果，通过 Gather Selection（GS）完成
   resident hit/miss、DRAM -> HBM 加载、resident 状态更新和 SFA indices 生成。
5. SFA 使用 resident block table、resident sequence length 和 resident logical indices
   完成稀疏 attention。
6. 保持请求状态机、prefill/decode phase barrier、finish/preempt 资源生命周期以及
   row-mode ACL Graph 语义。
7. 首期模型范围严格保持源实现：只支持 `DeepseekV32ForCausalLM`。

GLM-5/GLM-5.1 适配不属于首期迁移。首期不修改 GLM 模型识别、Indexer 行为、RoPE
处理、lightning indexer 分支或部署配置；待 DeepSeek-V3.2 等价迁移闭环后另行设计。

### 1.2 基线

| 项目 | 分支/提交 | 基线 |
| --- | --- | --- |
| 源仓库 | `vllm019-DSA-offload:tmp-opt`，HEAD `ad428d7bf6e481a8be2141c94dc35ffd31bcbfae` | `vllm-ascend-v0.19.1rc1-base`，commit `da421afad7192dac64e39ae1d32305d57344f3cf` |
| 目标仓库 | `vllm-ascend:dev_framework_adapt`，HEAD `57cbc0528eafb3e4c25099f7ef9de9e54af3e2de` | G2.5 v0.18 基线，配套 vLLM tag `v0.18.0` |

这是一项从 v0.19.1rc1 向 v0.18.0 的反向移植。不能直接 cherry-pick 源分支的
monkey patches；所有 vLLM 核心接口必须按 v0.18.0 的实际签名和对象生命周期重接。

目标分支没有在 G2.5 v0.18 基线上增加任何提交或源码修改：

- `dev_framework_adapt`
- `g2.5_base_v0.18`
- `work/g2.5_base_v0.18`
- `work/releases/v0.18.0`

以上引用均指向 `57cbc0528eafb3e4c25099f7ef9de9e54af3e2de`，分支间 ahead/behind
为 `0/0`，源码 diff 为空。因此本文所称“目标结构”全部是 G2.5 v0.18 原始基线结构，
不存在需要保留或兼容的 `dev_framework_adapt` 分支增量。迁移的唯一增量来源是
`vllm019-DSA-offload` 的 DSA Sparse 修改。

### 1.3 约束

- 不在本地运行端到端验证。
- 不编写或运行脚本做静态测试；静态检查只依赖逐文件代码阅读和差异复核。
- 不新增 dense/reference/旧 offload manager 等备用执行路径。
- 不增加跨 v0.18/v0.19 的版本分支，不通过 `hasattr/getattr(default)` 猜测接口。
- 配置启用后只允许完整 DSA 路径。必要前置条件不满足时直接终止初始化或调度。
- 保留源实现本身定义的 eager/graph 边界，例如 ENTER_SPARSE、full block dump 和
  multi-token forward 本来就不进入 row-mode graph；这些不是新增 fallback。
- 首期只接 `model_runner_v1`。`VLLM_USE_V2_MODEL_RUNNER` 启用时直接拒绝 DSA 配置，
  不维护第二套未完成的 v2 实现。
- 不支持 PCP、spec decode、P/D disaggregation、Sparse C8、非默认
  Scheduler 与 DSA 同时启用；迁移阶段不为这些组合添加兼容分支。

### 1.4 最小影响原则

1. 只迁移源分支为 DSA Sparse 新增的能力，不顺带重构 G2.5 v0.18 基线的 KV、SFA、
   graph 或 worker 框架。
2. 以目标文件为基准做局部修改，不用源分支版本整体替换
   `model_runner_v1.py`、`sfa_v1.py`、`worker.py`、`platform.py`。
3. 优先复用 G2.5 v0.18 基线已有的 `MultiGroupBlockTable`、per-group attention metadata、
   CpuGpuBuffer 和 KV tensor 初始化流程，只补它们缺少的 split-plane 语义。
4. DSA 未启用时不改变对象类型、调度顺序、cache layout、graph key 和模型行为。
5. 不新增通用模型抽象。首期沿用源实现的 `DeepseekV32IndexerCache` 和
   `DeepseekV32ForCausalLM` 定向接入。
6. 完整保留源框架已有的 DSA trace 能力及其受开关保护的 tensor 采样/统计逻辑；
   不新增额外 trace point、无条件日志、无条件 device-host 同步或通用 tracing 框架。
7. 不复制 v0.19.1 的跨版本兼容探测和与目标基线无关的状态分支，只保留功能必需的
   v0.18 确定性接口。

## 2. 源框架必须保持的架构

### 2.1 三个 KV 平面

```text
Indexer HBM plane
  完整原始序列；原始 positions；独立 IndexerKVSpec/group/block table
                         |
                         | 每层 lightning indexer topK
                         v
MLA resident HBM plane <- GS <- MLA hot DRAM plane
  budget + tail                 已完成 full blocks，按 layer 存储
  resident positions            original logical block -> DRAM pool block
                         |
                         v
                  SFA resident indices
```

三个平面的所有权不能合并：

- Indexer HBM block 数随完整上下文增长。
- MLA resident HBM block 数按活动请求的 `budget + tail` 规划。
- DRAM hot store 保存 MLA full block，是被 GS 回源的数据面。

### 2.2 请求状态机

```text
PREFILL -> DENSE_DECODE -> ENTER_SPARSE_DECODE -> SPARSE_DECODE
```

- `PREFILL`：Indexer 和 MLA 都按完整序列写入。
- `DENSE_DECODE`：未达到阈值，仍用完整 MLA layout。
- `ENTER_SPARSE_DECODE`：首次建立 resident layout，必须走 eager。
- `SPARSE_DECODE`：每层执行 Indexer -> GS -> resident SFA。

稀疏阈值保持为 `round_up(hbm_sparse_budget, block_size) + block_size`。MLA resident
layout 保持 `[sparse budget slots][tail slots]`，不能改成静态 carve-out 或全局共享 slot。

### 2.3 每层计算顺序

```text
attention_begin
  -> 绑定本层 MLA/Indexer cache zone
  -> 校验进入 sparse 前的 full-block dump readiness
  -> 写本轮 MLA KV 和 Indexer K
  -> lightning indexer（完整 Indexer block table）
  -> after_indexer / Gather Selection
  -> SFA（resident MLA block table + resident indices）
  -> attention_finished
  -> dump 本层新完成的 MLA full blocks 到 DRAM
```

GS 必须按 layer 执行。topK 选择、resident status 和 DRAM -> HBM 加载都是 token-layer
粒度；不能把某层 topK 复用到其他层。

### 2.4 Trace 语义

首期等价迁移以下正式 trace 功能：

- `lightning_indexer`：记录 Indexer 输入、block table、candidate length 和 topK 摘要。
- `gather_selection`：记录 GS 输入输出 tensor 摘要。
- `gather_selection_stats`：记录 resident hit/miss/overlap 统计。
- `ranks`、`layers` 过滤，以及 `sync` 可选同步。

实现来源为 `vllm_ascend/dsa_sparse/dsa_trace.py`，调用点位于
`vllm_ascend/attention/sfa_v1.py` 和
`vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py`。该功能默认关闭；只有用户显式配置
trace point 时才执行 tensor sample、min/max、统计和可选 NPU stream synchronize。

“最小化影响”只表示不扩展这套 trace，不表示删除它。迁移后配置字段、三个 trace
point、过滤规则、日志内容和同步开关均保持源实现语义。

## 3. v0.19.1rc1 与 G2.5 v0.18 基线差异

本章分析的是源 DSA 所依赖的 v0.19.1rc1 接口与目标 G2.5 v0.18 原始接口之间的版本
差异，不是 `dev_framework_adapt` 相对 G2.5 基线的分支差异。

### 3.1 vLLM KV 管理差异

v0.18.0 的 `KVCacheCoordinator` 在 `vllm/v1/core/kv_cache_coordinator.py:28-69`
只创建一个共享 `BlockPool`，所有 `single_type_managers` 都引用该 pool；分配接口在
`:71-176` 仍以各 group 所需 block 数求和。源 DSA 则要求 Indexer 和 MLA 使用不同容量
的物理 pool。

适配方案：

1. 以 v0.18.0 的 `BlockPool` 字段和方法为模板迁入 `MultiBlockPool`，内部为每个
   `kv_cache_group_id` 创建独立 pool。
2. 按 v0.18.0 的 manager 方法签名重写 group routing，不复制 v0.19.1 的方法签名。
3. 重写 coordinator 初始化，使 manager 在构造时拿到对应 group pool。
4. `get_num_blocks_to_allocate`、`allocate_new_computed_blocks`、`allocate_new_blocks`、
   `free`、`touch` 和 prefix-hit 路径都必须携带确定的 group id。
5. 为 v0.18.0 的 `KVCacheManager` 补入 DSA 所需的 per-group admission 计算，不依赖
   v0.19.1 才新增的 `KVCacheManager.can_fit_full_sequence()`。
6. 删除源 patch 对 v0.19.1 `scheduler_reserve_full_isl` 和
   `can_fit_full_sequence()` 的依赖；按 v0.18.0 Scheduler 已有 token budget、
   chunked-prefill 和 waiting admission 流程计算本轮可分配量。
7. 不保留共享 pool 分支。启用 DSA 时 coordinator 必须使用 multi-pool。

关键目标位置：

- `vllm/v1/core/block_pool.py:129-500`
- `vllm/v1/core/kv_cache_coordinator.py:28-230,547-591`
- `vllm/v1/core/kv_cache_manager.py:106-390`
- `vllm/v1/core/single_type_kv_cache_manager.py`

### 3.2 KV spec、group 和容量计算差异

目标 `vllm-ascend` 已用 `AscendMLAAttentionSpec` 表示组合的 MLA + Indexer cache，位置为
`vllm_ascend/patch/platform/patch_kv_cache_interface.py:13-151`。目标 ModelRunner 在
`vllm_ascend/worker/model_runner_v1.py:3254-3320` 对 DSA 层只生成一个组合 spec。

适配方案：

1. 增加独立 `IndexerKVSpec`，只描述完整 Indexer cache。
2. DSA 启用时，MLA spec 的 `head_size` 只包含 noPE/RoPE MLA resident payload；
   Indexer 不再作为该 tensor 的第三段。
3. 只 patch `deepseek_v2.py` 中 `DeepseekV32IndexerCache.get_kv_cache_spec()`，使其
   在 DeepSeek-V3.2 DSA 开启时返回 `IndexerKVSpec`。该类位于配套 vLLM v0.18.0 的
   `vllm/model_executor/models/deepseek_v2.py:584`。不扩展该 patch 的模型范围。
4. 修改 v0.18.0 `get_kv_cache_groups()`，只按明确的 spec 类型拆成 Indexer group 和
   MLA resident group。
5. 修改 `get_kv_cache_config_from_groups()`，为每个 group 记录确定的 block 数，并按该
   group 容量生成 `KVCacheTensor.size`。
6. 修改 v0.18.0 `get_kv_cache_configs()` 的跨 worker 最小容量归一逻辑。不能继续用
   单一 `min_num_blocks` 等比例收缩两组 tensor，必须分别归一 Indexer 与 MLA group。
7. 内存上界保持：Indexer 按 `max_model_len`；MLA 按每请求 `sparse budget + one tail
   block`；实际 HBM 按 `indexer_mla_block_ratio` 分配。

目标落点：

- `vllm_ascend/patch/platform/patch_kv_cache_interface.py`
- 新增 `vllm_ascend/patch/dsa_sparse/patch_deepseek_v2.py`
- 新增 `vllm_ascend/patch/dsa_sparse/patch_kv_cache_utils.py`
- `vllm_ascend/worker/model_runner_v1.py:3254-3320`
- vLLM v0.18.0 `vllm/v1/core/kv_cache_utils.py:1467-1607`

### 3.3 Scheduler 与 SchedulerOutput 差异

v0.18.0 Scheduler 主流程位于 `vllm/v1/core/sched/scheduler.py:338`，preempt、请求状态
更新和释放分别位于 `:929`、`:1275`、`:1705-1813`。G2.5 v0.18 基线的 Scheduler
包含 `skipped_waiting` 语义，因此源 `patch_scheduler.py` 不能覆盖或假定 waiting queue
布局。

适配方案：

1. 保留源实现的 prefill/decode phase barrier，但以 v0.18.0
   `_select_waiting_queue_for_scheduling()` 和 `skipped_waiting` 语义实现暂存与恢复。
2. 包装 v0.18.0 `KVCacheManager.allocate_slots()` 的原始签名，在调用前计算
   `ReqStage`、resident valid length 和本轮 MLA slots。
3. PREFILL 完成且各层 dump 已提交后释放 MLA full blocks，只保留 tail；Indexer blocks
   不释放。
4. preempt 清空 resident layer 状态和 dump readiness，但保留仍被请求引用的 DRAM
   backing；finish 同时释放 resident row 与 DRAM block refs。
5. 直接为 v0.18.0 的 `NewRequestData`、`CachedRequestData`、`SchedulerOutput` 定义 DSA
   字段，不使用运行时动态附加：
   - `block_hashes`
   - `req_dsa_stage`
   - `req_dsa_resident_valid_seq_len`
   - `req_dsa_sparse_budget_tokens`
6. `make_empty()`、`from_request()` 和 Scheduler 构造输出的位置一并更新，保证 EngineCore
   到 worker 的序列化结构唯一。

目标落点：

- 新增 `vllm_ascend/patch/dsa_sparse/patch_request.py`
- 新增 `vllm_ascend/patch/dsa_sparse/patch_scheduler_output.py`
- 新增 `vllm_ascend/patch/dsa_sparse/patch_scheduler.py`
- vLLM v0.18.0 `vllm/v1/core/sched/output.py:31-256`
- vLLM v0.18.0 `vllm/v1/core/sched/scheduler.py:338-1087,1275-1813`

### 3.4 ModelRunner 和 block table 差异

目标 `model_runner_v1` 已原生按多个 KV group 构建 metadata：

- `_prepare_inputs()`：`vllm_ascend/worker/model_runner_v1.py:567-614`
- `_build_attention_metadata()`：`:2034-2235`
- KV tensor 初始化：`:2629-3051`
- `get_kv_cache_spec()`：`:3254-3320`

这部分应作为迁移主落点，而不是覆盖成源分支的完整文件。

适配方案：

1. 在 request state 中保存 group-ordered block ids 和 block hashes，group 顺序由
   `KVCacheConfig.kv_cache_groups` 唯一确定。
2. `_prepare_inputs()` 沿用 v0.18.0 的 NumPy/CpuGpuBuffer 流程，同时生成：
   - 原始 positions：Indexer group 使用；
   - resident positions：MLA group 使用；
   - 每组独立 slot mapping。
3. 直接按 group 取得底层 `BlockTable`，分别调用其
   `compute_slot_mapping(req_indices, indexer_positions/resident_positions)`，再统一 commit；
   禁止先生成 group 0 再复制到其他 group，也不照搬 v0.19 的 GPU positions 实现。
4. `_build_attention_metadata()` 按 group 取得 block table/slot mapping，并把 Indexer
   metadata 与 MLA metadata 通过明确字段关联。不得在 SFA 内找不到 Indexer metadata
   时退回 MLA block table。
5. 在 model forward 前调用 `build_dsa_meta()` 和 `execute_begin()`；forward 成功结束后
   调用 `execute_finished()`；异常由上层结束当前执行，不增加清空后继续执行的路径。
6. KV tensor 初始化按各 group 的 `dsa_num_blocks` 分配：Indexer tensor 与 MLA
   noPE/RoPE tensors 分开；不再生成目标基线原有的三元组合 cache tuple。
7. worker 初始化只接受 `DeepseekV32ForCausalLM`，创建唯一的
   `DSASparseV1` worker manager 并注入 runner。
8. `dsa_model_runner_state.update_states()` 以目标 v0.18.0
   `GPUModelRunner._update_states` 为母版做窄幅 DSA override：Indexer group 继续 append
   dense blocks，MLA group 在 sparse cached request 上用目标 `BlockTable.add_row()` 替换
   resident row；不移植 v0.19 的 async-spec deferred correction 和不存在的 `reset_row()`。

目标文件：

- `vllm_ascend/worker/block_table.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/worker/worker.py:318-324,327-525`
- 新增 `vllm_ascend/dsa_sparse/dsa_model_runner_state.py`

### 3.5 SFA 接入差异

目标 SFA 的 Indexer 主流程位于 `vllm_ascend/attention/sfa_v1.py:877-1010`。该文件还
包含 GLM 专用 RoPE 和 lightning indexer 分支，但首期迁移不进入、不修改这些分支。

适配方案：

1. 只在 DeepSeek-V3.2 的 Indexer 之后、SFA 之前插入 DSA hook；目标文件中 GLM 专用
   RoPE、`torch_npu.npu_lightning_indexer` 和量化判断代码保持不变。
2. 扩展 `AscendSFAMetadata`，明确保存：
   - `indexer_block_table`、`indexer_slot_mapping`、原始 `seq_lens/positions`；
   - `resident_block_table`、`resident_slot_mapping`、resident `seq_lens/positions`；
   - full-batch row mode、query row 和 GS 输出 indices。
3. Indexer 写 cache 和计算 topK 始终使用 Indexer group metadata。
4. `after_indexer()` 接收 DeepSeek-V3.2 lightning indexer 的 topK tensor，保持源实现
   layout，仅在 `AscendDSAOpsBackend` 的唯一边界规范化为 GS ABI。
5. SFA 只使用 GS 输出的 resident logical indices、MLA resident block table 和 resident
   valid length；禁止继续使用原始 topK 或完整 Indexer block table。
6. `ops/mla.py` 在 MLA forward 外围加入 `attention_begin/attention_finished`，并让
   `IndexerWrapper` 暴露独立 Indexer cache layer name。
7. 保留 v0.18.0 的 `forward_context.virtual_engine`，从当前 virtual engine 对应的
   MLA/Indexer cache tensor 建立 layer cache zone；不照搬 v0.19 删除该参数后的选择逻辑。
8. 保留目标 `AscendSFABackend.get_kv_cache_shape()` 的四参数契约，不复制源分支的
   `cache_type` 第五参数；split plane 由 KV spec/group 决定，不由 backend shape 参数决定。

目标文件：

- `vllm_ascend/attention/sfa_v1.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/ops/mla.py`

### 3.6 EngineCore、patch 启动与多进程差异

v0.18.0 与 v0.19.1 的 EngineCore 入口名称基本一致，但实现行号、进程管理参数和
`core_client` 调用链不同。目标 v0.18.0 位置包括：

- `vllm/v1/engine/core.py:227`：`_initialize_kv_caches`
- `vllm/v1/engine/core.py:776,1029`：`EngineCoreProc/run_engine_core`
- `vllm/v1/engine/utils.py:81,843`：进程 manager 与 launch
- `vllm/v1/engine/core_client.py:79`：client 创建

适配方案：

1. general plugin 在每个解释器安装一次固定的 v0.18.0 patches。
2. Platform 配置阶段在 EngineCore spawn 前物化 DSA config。
3. 按 v0.18.0 的实际入口重写 EngineCore 子进程 bootstrap；不复制 v0.19.1 的参数
   探测和多种入口 fallback。
4. 修正 EngineCore 内部对 `get_kv_cache_configs` 的 by-value import，确保 scheduler 和
   worker 得到同一 split group config。
5. 初始化 request block hasher，SchedulerOutput 的 block hashes 作为 DRAM block 复用
   的唯一 hash 来源。

### 3.7 ACL Graph 差异

源实现给 `BatchDescriptor` 增加 `dsa_graph_phase` 并扩展 dispatcher key。v0.18.0 的
`BatchDescriptor`/`CudagraphDispatcher` 必须重新按当前字段顺序和 dispatch 签名接入。

适配方案：

1. 先完成 eager 完整链路，再迁 `dsa_graph_gate.py` 和 `dsa_graph_buffers.py`。
2. 仅支持源实现已有的 single-token row-mode decode graph。
3. graph buffer 固定保存 row mode、candidate length、两张 block table、resident status
   view 和 attention indices；capture/replay 不重新分配 tensor。
4. ENTER_SPARSE、full block dump、multi-token、PCP 路径固定为 eager，不新增其他 graph
   key 或自动降级分支。
5. DSA 配置与 `enable_npugraph_ex` 互斥，使用 ACL FULL graph 路径。

## 4. 迁移后的模块边界

### 4.1 原样迁移后仅做 import/API 调整的核心模块

以下模块的数据结构和算法应保持源实现，不做第二套抽象：

- `vllm_ascend/dsa_sparse/dsa_types.py`
- `vllm_ascend/dsa_sparse/dsa_req_meta.py`
- `vllm_ascend/dsa_sparse/dsa_forward_batch.py`
- `vllm_ascend/dsa_sparse/dsa_batch_tensor_utils.py`
- `vllm_ascend/dsa_sparse/dsa_resident_pool.py`
- `vllm_ascend/dsa_sparse/dsa_hot_kv_store_core.py`
- `vllm_ascend/dsa_sparse/dsa_ascend_hot_kv_store.py`
- `vllm_ascend/dsa_sparse/dsa_layer_cache_zones.py`
- `vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py`
- `vllm_ascend/dsa_sparse/dsa_trace.py`

### 4.2 必须按目标基线重写接入的模块

- `dsa_config.py`：固定 v0.18.0 配置入口和互斥项。
- `dsa_attention_layout.py`：读取目标 `AscendSFAMetadata` 的显式双平面字段。
- `dsa_model_runner_state.py`：适配 v0.18.0 request state 和 group block ids。
- `dsa_sparse.py`：适配目标 KV group/config 和 DeepSeek-V3.2 layer hook。
- `dsa_graph_gate.py`、`dsa_graph_buffers.py`：适配 v0.18.0 graph API。
- `patch/dsa_sparse/*`：所有 vLLM monkey patch 按 v0.18.0 重写。
- `worker/model_runner_v1.py`、`attention/sfa_v1.py`：以目标文件为基准做局部接入。

### 4.3 不迁移的旧 ASU 框架

不得把 `feat/kv-offload-v011-compact-sfa` 的以下机制并入 DSA：

- `OffloadKVCacheV0Manager` resident 状态机；
- HBM tensor 尾部静态 carve-out；
- MicroKV token record 作为 DSA backing store；
- lookup/maintain 两次 token -> slot 转换；
- resolved-slot SFA 的第二套 block table resolver。

可参考的仅是 request finish 接入位置、内存预算入口和少量纯 block/slot 换算语义。
迁移后 `DSAResidentTokenPool + GS` 是 resident 映射唯一真源，`DSAHotKVStore` 是 MLA
offload backing 唯一真源。

## 5. 实施阶段

### 阶段 0：冻结目标契约

1. 固定目标提交 `57cbc0528e` 和配套 vLLM `v0.18.0`。
2. 记录以下实际签名：KV spec/config、BlockPool、SingleType manager、Coordinator、
   Scheduler、SchedulerOutput、ModelRunner、SFA metadata 和 graph dispatcher。
3. 固定首期能力矩阵：DeepSeek-V3.2 BF16、默认 Scheduler、model_runner_v1、single-token
   decode、无 PCP/spec decode/P-D/Sparse C8；保留源框架依赖 block hash 的 prefix/full
   block 复用语义。
4. 确认 `DeepseekV32ForCausalLM` 的 Indexer cache 层名和每层 MLA cache layout。

完成条件：迁移文档中的所有目标接口都能在固定 commit 找到唯一代码位置。

### 阶段 1：算子和纯 DSA 数据面

1. 迁移 `csrc/gather_selection_kv_cache/**`。
2. 更新 `csrc/build_aclnn.sh`、`csrc/torch_binding.cpp` 和
   `csrc/torch_binding_meta.cpp`，注册 PrivateUse1 与 Meta schema。
3. 迁移 resident pool、hot store、layer cache registry、batch tensor utilities 和
   Ascend ops backend。
4. 迁移 `dsa_trace.py` 以及 SFA/GS backend 中受 trace 开关保护的采样和统计调用点。
5. 保持 GS ABI、row mode 值、status layout 和 attention indices layout 不变。

完成条件：通过逐文件检查确认 Python 调用参数顺序、Torch schema、host op def、tiling
参数和 kernel 参数完全一致。此阶段不执行算子。

### 阶段 2：split KV spec、容量与 allocator

1. 加入 `IndexerKVSpec` 和 DeepSeek-V3.2 Indexer spec patch。
2. 修改 ModelRunner 的 KV spec，拆出 Indexer/MLA 两组。
3. 重写 v0.18.0 KV group/config 生成和 per-group 内存归一。
4. 迁移 `MultiBlockPool`、Indexer manager 和 coordinator group routing。
5. 明确两组 block 数、tensor size、group id 和 block table 的一一对应关系。

完成条件：仅通过代码阅读能够从 `get_kv_cache_spec()` 追踪到最终两个 HBM tensor
平面及其各自 block 数；任一处不再假定所有 group 共用 `KVCacheConfig.num_blocks`。

### 阶段 3：Scheduler 控制面

1. 迁移 `ReqStage` 和 request 字段。
2. 迁移 phase barrier、resident slot 规划和 allocate wrapper。
3. 迁移 SchedulerOutput DSA metadata 与 block hashes。
4. 接通 PREFILL full-block 释放、preempt 和 finish 生命周期。
5. 对 v0.18.0 `skipped_waiting` 行为逐分支复核，保证被暂存请求不丢失、不重复入队。

完成条件：对 PREFILL、DENSE_DECODE、ENTER_SPARSE、SPARSE、preempt、finish 六条路径
逐条画出 block 所有权变化，Indexer 与 MLA 的释放行为无歧义。

### 阶段 4：Worker、ModelRunner 与 DRAM dump

1. 仅为 `DeepseekV32ForCausalLM` 创建 worker DSA manager。
2. 迁移 request state update、block hashes、resident pool row 生命周期。
3. 以 v0.18.0 `_update_states` 为母版适配 per-group block row，并用目标
   NumPy/CpuGpuBuffer 方式生成 positions、slot mapping 和 block table。
4. 按 split config 分配 Indexer 与 MLA cache tensors。
5. 接通 `build_dsa_meta/execute_begin/execute_finished`。
6. 接通逐层 `attention_begin/attention_finished` 与 full block dump readiness。

完成条件：从 SchedulerOutput 到 layer runtime batch 的每个 request row、group id、block
id、position 和 pool row 均有唯一来源，finish/preempt 后无悬挂所有权。

### 阶段 5：DeepSeek-V3.2 Indexer -> GS -> SFA

1. 保留目标 DeepSeek-V3.2 现有 Indexer 算法和 topK layout，不修改 GLM 分支。
2. 为 SFA metadata 增加显式 Indexer/MLA resident 双平面字段。
3. Indexer 写入和 topK 使用完整 Indexer plane。
4. 在 indexer post-process 后调用 GS。
5. 用 GS resident indices 替换 SFA indices，并切换到 MLA resident block table/seq len。
6. 逐行核对 dense、sparse、PAD row 的输入和值域。

完成条件：代码中不存在 DSA 启用后将 MLA block table 作为 Indexer table、将原始 topK
直接传给 resident SFA，或缺少 GS 结果时继续执行 SFA 的分支。

### 阶段 6：row-mode ACL Graph

1. 适配 v0.18.0 BatchDescriptor 和 dispatcher。
2. 迁移 gate、capture dummy batch、stable buffers 和 replay copy。
3. 把 graph begin/end 接到目标 ModelRunner 的 capture/replay 生命周期。
4. 复核所有会被 GS 原址修改的 tensor 地址在 capture/replay 期间稳定。

完成条件：graph 路径与 eager 路径共享同一 DSA forward/layer batch 语义，只差稳定 buffer；
不出现第二套 GS 或 SFA 参数构造逻辑。

### 阶段 7：只读复核与交付

本地无运行环境，本阶段只做代码阅读复核：

1. 按启动、KV config、调度、worker state、逐层计算、释放六条调用链逐文件走查。
2. 对照 GS schema 检查所有 tensor dtype、rank、layout 和编号空间。
3. 对照迁移前后的 G2.5 v0.18 源码检查非 DSA 路径和 GLM 代码未发生行为变化。
4. 检查 DSA 配置开启时不存在旧 combined cache、旧 compact manager 或静默 fallback。
5. 记录未执行项：C++ 编译、算子执行、单元测试、端到端精度、性能和 graph replay。

## 6. 关键数据契约

| 数据 | 语义 | 生产者 | 消费者 |
| --- | --- | --- | --- |
| Indexer block table | 原始完整序列逻辑块 -> Indexer HBM block | Scheduler/MultiBlockPool | lightning indexer |
| MLA resident block table | resident logical block -> MLA HBM block | Scheduler/MultiBlockPool | GS、SFA |
| DRAM logical block table | 原始 logical block -> host arena pool block | DSAHotKVStore | GS |
| original positions | 请求真实 token position | ModelRunner | Indexer cache write |
| resident positions | budget/tail 平面的 slot position | DSA plan/ModelRunner | MLA cache write |
| topK indices | 原始完整序列 token/segment id | lightning indexer | GS |
| resident status | resident slot -> 原始 token/segment id | GS，持久化于 DSAResidentTokenPool | 下一层/下一 step GS |
| SFA indices | resident logical slot | GS | SFA |
| block hashes | 原始 full block 内容 hash | Scheduler | DSAHotKVStore 引用复用 |

禁止混用的编号空间：

- 原始 token position；
- Indexer HBM physical block id；
- MLA resident HBM physical block id；
- resident logical slot；
- DRAM arena pool block id；
- request/resident pool row。

## 7. 预计源码修改清单

### 7.1 新增/迁移

- `csrc/gather_selection_kv_cache/**`
- `vllm_ascend/dsa_sparse/**`
- `vllm_ascend/patch/dsa_sparse/**`
- `vllm_ascend/patch/platform/patch_dsa_sparse.py`
- `vllm_ascend/patch/worker/patch_dsa_sparse.py`

### 7.2 修改

- `setup.py`
- `csrc/build_aclnn.sh`
- `csrc/torch_binding.cpp`
- `csrc/torch_binding_meta.cpp`
- `vllm_ascend/__init__.py`
- `vllm_ascend/platform.py`
- `vllm_ascend/utils.py`
- `vllm_ascend/patch/__init__.py`
- `vllm_ascend/patch/platform/__init__.py`
- `vllm_ascend/patch/platform/patch_kv_cache_interface.py`
- `vllm_ascend/patch/worker/__init__.py`
- `vllm_ascend/worker/block_table.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/worker/worker.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/attention/sfa_v1.py`
- `vllm_ascend/ops/mla.py`

实际实现不直接修改 vLLM 仓库；上文列出的 vLLM 文件是 monkey patch 的目标契约和
代码阅读基线。

## 8. 风险与处理原则

| 风险 | 后果 | 处理 |
| --- | --- | --- |
| v0.19 patch 直接套到 v0.18 | 签名错位或绕过 `skipped_waiting` | 每个 wrapper 以 v0.18 源码重写 |
| 两个 KV group 仍共享 pool | Indexer/MLA 容量互相挤占，释放错误 | DSA 启用后只用 MultiBlockPool |
| 整体替换目标 SFA | 覆盖目标已有模型修复 | 以目标 SFA 为基线，只插入 DeepSeek DSA hook |
| combined cache tuple 残留 | Indexer 与 MLA block 数无法独立 | DSA 路径只接受 split tensors |
| prefill block 先释放后 dump | DRAM backing 缺数据 | dump readiness 先于 Scheduler 释放 |
| finish/preempt 混同 | 泄漏或提前释放 DRAM refs | 保持两条独立生命周期 |
| graph tensor 地址变化 | replay 读写错误 | graph stable buffers 持有所有副作用 tensor |
| v2 runner 被误启用 | manager hook 和 metadata 缺失 | 初始化时直接拒绝该组合 |
| 引入 v0.1.1 compact manager | 双 resident 真源、两次坐标转换 | 明确不迁移该框架 |

## 9. 后续有环境时的验收边界

本迁移计划不在当前本地执行验证。具备可运行环境后，最低验收应覆盖：

1. DeepSeek-V3.2 BF16 长上下文从 dense 进入 sparse 的输出精度。
2. pure dense、ENTER_SPARSE、steady sparse 和 mixed dense/sparse batch。
3. 多请求 continuous batching、request finish 和 preempt/recompute。
4. 每层 full-block dump、DRAM hash/refcount 复用和 GS hit/miss。
5. eager 与 row-mode ACL Graph 的结果一致性。
6. HBM 中 Indexer 容量随完整上下文增长、MLA 容量受 sparse budget 约束。

在这些验收完成前，只能声明代码迁移和静态调用链复核完成，不能声明功能或性能验证完成。
