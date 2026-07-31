# vllm019-DSA-offload 分支修改分析

## 1. 分析范围

- 仓库：`vllm019-DSA-offload`
- 当前分支：`tmp-opt`
- 分析基线：`da421afad7192dac64e39ae1d32305d57344f3cf`
- 当前 HEAD：`ad428d7bf6e481a8be2141c94dc35ffd31bcbfae`
- 基线之后提交数：11
- 最终差异：74 个文件，新增 17,863 行，删除 287 行
- 非文档源码文件：67 个

本文分析 `da421afad7192dac64e39ae1d32305d57344f3cf..HEAD` 范围内的最终修改，重点说明 DSA 稀疏卸载的代码框架、初始化和运行流程、主要修改点以及当前限制。源码文件清单不包含 `README*`、`docs/` 和示例目录中的说明文档。

## 2. 修改目标

该分支在 vLLM-Ascend 中实现了一套面向 DeepSeek-V3.2 的 DSA 稀疏 KV Cache 卸载原型，目标如下：

1. Indexer KV 全量保留在 HBM，保证 lightning-indexer 能扫描完整上下文。
2. MLA/full KV 在 HBM 中只保留固定 sparse budget 和未满尾块。
3. 其余 MLA 满块卸载到 worker 本地 DRAM hot store。
4. 每个 decode step 由 lightning-indexer 选择 topK token。
5. 使用 row-mode `gather_selection_kv_cache` 完成 resident 命中判断、DRAM KV 加载、resident slot 更新和 SFA attention indices 生成。
6. dense、sparse、padding 请求使用同一条 full-batch 路径。
7. 稳态 single-token decode 支持 ACL Graph capture/replay。
8. 尽量不修改上游 vLLM，通过 vLLM-Ascend plugin 和 runtime patch 集成。

## 3. 整体代码框架

### 3.1 配置和 Patch 启动层

用户通过 `additional_config["dsa_sparse_config"]` 传入 DSA 参数。`dsa_config.py` 将公开参数转换成 `CacheConfig` 动态属性，并处理以下内容：

- DSA 总开关和 split indexer cache 开关。
- HBM sparse budget，默认 2048 token。
- Indexer/MLA block ratio，默认 3:1。
- 最大活动请求数和 DRAM hot block 倍率。
- row-mode decode graph 开关。
- trace point、rank、layer 和同步选项。
- 图模式与 `enable_npugraph_ex` 的冲突校验。

`setup.py` 增加 `vllm.general_plugins` 入口 `ascend_dsa_sparse`。`patch_runtime.py` 统一安装 KV cache interface、SchedulerOutput、KV cache 解耦、CUDAGraph phase、DeepSeek indexer spec、Request、Scheduler 和 EngineCore patch。

由于 EngineCore 和 worker 可能由新解释器启动，`patch_engine_process.py` 会在 EngineCore 进程边界重新附加配置、安装 patch 并验证关键 callable 是否仍指向 DSA wrapper，避免 import cache 或其他 patch 覆盖 DSA hook。

### 3.2 KV Cache 解耦层

原有统一 KV block pool 被拆成两个 HBM plane：

- `IndexerKVSpec`：保存完整上下文，供 lightning-indexer 使用。
- MLA/full resident spec：只保存 sparse budget 和当前未满尾块，供 SFA 使用。

同时增加 worker-local DRAM hot store，保存从 MLA plane 卸载的完整 KV block。

`MultiBlockPool` 为不同 KV group 维护独立物理 block pool，解决 Indexer 和 MLA block 数量、容量及生命周期不同的问题，同时继续共享 block hash 和 cache event 等公共结构。

容量分配由 `patch_kv_cache_utils.py` 完成。它按 KV page size、layer 数和 `indexer_mla_block_ratio` 计算两组 block 数量，并输出 DSA HBM cache capacity report。开启 DSA 后，如果没有生成 Indexer/MLA 两个 group，相关路径会 fail-fast。

### 3.3 Scheduler 管控层

Scheduler 为每个请求维护以下状态机：

```text
PREFILL
  -> DENSE_DECODE
  -> ENTER_SPARSE_DECODE
  -> SPARSE_DECODE
```

- `PREFILL`：prompt 或 chunk prefill，完整写入 cache。
- `DENSE_DECODE`：已经进入 decode，但上下文还未超过稀疏阈值。
- `ENTER_SPARSE_DECODE`：首次达到稀疏条件，需要初始化 resident layout。
- `SPARSE_DECODE`：稳态稀疏 decode。

Scheduler patch 负责：

- 推进请求 `ReqStage`。
- 估算本轮 MLA resident slots。
- 包装 `KVCacheManager.allocate_slots()`。
- 在长 prompt prefill 完成后释放 MLA 满块，只保留未满尾块。
- 保证 prefill/decode phase barrier。
- 将 DSA metadata 放入 SchedulerOutput。

SchedulerOutput 新增：

- `req_dsa_stage`
- `req_dsa_resident_valid_seq_len`
- `req_dsa_sparse_budget_tokens`
- new/cached request 的 `block_hashes`

### 3.4 Worker 和元数据层

Worker 仅针对 `DeepseekV32ForCausalLM` 创建 `DSASparseV1`，并注入：

- `AscendDSAOpsBackend`
- `AscendDSAHotKVStore`
- `DSAResidentTokenPool`

`NPUModelRunner.prepare_dsa_scheduled_request()` 在 model forward 前调用：

```text
build_dsa_meta()
  -> 可选 graph replay batch 准备
  -> execute_begin()
```

元数据按生命周期拆分：

- 请求级：`ReqMeta`、`ReqForwardPlan`、`ReqSparseDecodeForwardPlan`。
- forward 级：`DSAModelForwardMeta`、`DSAForwardSparseDecodeBatch`。
- layer 级：`DSALayerRuntimeBatch`、`DSAForwardLayerBatch`、`DSALayerSparseDecodeBatch`。
- 持久资源：resident pool、DRAM hot store、layer cache registry、graph stable buffers。

`build_dsa_meta()` 只组装请求 block table、query range、resident pool row 和 full-batch tensor，不执行 indexer、KV 搬运或 resident 替换。这些动作被推迟到 layer attention hook。

### 3.5 Attention 和自定义算子层

每层 MLA attention 增加三个 hook：

1. `attention_begin`
2. lightning-indexer 之后的 `after_indexer`
3. `attention_finished`

`sfa_v1.py` 将 dense、sparse 和 padding row 统一为 full-batch lightning-indexer 和 GS 路径。每行执行模式为：

- `PAD = 0`
- `DENSE = 1`
- `SPARSE = 2`

GS 不再要求 Python 将 batch 拆成 dense/sparse 子 batch，而是在 kernel 内按 row mode 处理差异。

`gather_selection_kv_cache` 完成两类状态更新：

1. 原址更新 `resident_slot_token_status`，记录 resident slot 当前承载的原始 token/segment。
2. 输出 SFA 使用的 `attention_indices`。对 sparse row，索引表示 resident logical slot，而不是原始 token id。

### 3.6 Resident Pool 和 DRAM Hot Store

`DSAResidentTokenPool` 预分配固定五维状态 tensor：

```text
[layer, request_pool_idx, 1, 1, resident_slot]
```

它统一管理 request pool row、每层 resident count 和 resident slot 到原 token 的映射。固定分配减少逐层动态创建，也为图模式提供稳定 tensor 地址。

`DSAHotKVStore` 使用 host arena、逻辑 DRAM block table、ready table 和引用计数保存卸载的满块。request 完成或被抢占时，worker 会同时释放 resident pool row 和 DRAM block 引用。

### 3.7 ACL 图模式层

图模式只覆盖 shape、地址和副作用稳定的 single-token decode：

- 支持 pure dense、pure sparse 和 dense/sparse mixed batch。
- `ENTER_SPARSE_DECODE` 走 eager。
- 发生新满块 dump 的 forward 走 eager。
- multi-token/spec decode 走 eager。
- capture size 不匹配时走 eager。

`dsa_graph_gate.py` 判断当前 batch 是否允许进入 DSA row-mode graph phase，`dsa_graph_buffers.py` 管理 capture/replay 所需的固定地址 block table、row mode、candidate lens 和 attention indices 等 tensor。

## 4. 运行时完整流程

```text
1. LLM 创建
   additional_config.dsa_sparse_config
        |
2. Ascend platform 配置检查
   配置映射 + general plugin + runtime patches
        |
3. KV cache 初始化
   IndexerKVSpec + MLA resident spec
   MultiBlockPool 分组分配
   DRAM hot store / resident pool 初始化
        |
4. Scheduler 调度
   推进 ReqStage
   估算 resident slots
   分配或释放 MLA HBM block
   输出 stage、budget、resident len、block hashes
        |
5. ModelRunner forward
   build_dsa_meta
   构造 full-batch row-mode tensor
   graph gate 判定
        |
6. 每层 attention_begin
   注册本层 cache
   dump 新满块到 DRAM
   构造 layer runtime batch
        |
7. lightning-indexer
   使用完整 Indexer HBM cache 计算 topK
        |
8. gather-selection
   检查 resident hit/miss
   miss KV 从 DRAM 加载到 HBM resident slot
   更新 resident slot status
   生成 SFA attention indices
        |
9. SFA attention
        |
10. attention_finished / execute_finished
    提交 resident metadata
    清理 layer/forward 临时状态
```

## 5. 主要修改点

下表中的行号基于当前 HEAD `ad428d7bf6e481a8be2141c94dc35ffd31bcbfae`。主实现位置用于定位修改入口，辅助位置用于追踪调用、资源管理或验证逻辑。

| 序号 | 修改点 | 主实现位置 | 辅助位置 |
| --- | --- | --- | --- |
| 1 | 新增 DSA general plugin 和多进程 patch bootstrap | `setup.py:544`；`vllm_ascend/__init__.py:31 register_dsa_sparse()`；`vllm_ascend/patch/dsa_sparse/patch_runtime.py:11 install_dsa_runtime_patches()` | `vllm_ascend/patch/dsa_sparse/patch_engine_process.py:121 _prepare_dsa_engine_bootstrap()`；同文件 `:139 _dsa_sparse_run_engine_core()` |
| 2 | 新增统一 `dsa_sparse_config` 配置入口、默认值和冲突校验 | `vllm_ascend/dsa_sparse/dsa_config.py:24 _DSA_SPARSE_CONFIG_FIELD_MAPPINGS`；`:76 _normalize_dsa_sparse_config()`；`:109 attach_dsa_sparse_cache_attrs()` | `vllm_ascend/platform.py:65 _ensure_dsa_sparse_runtime_patches()`；`vllm_ascend/worker/worker.py:102 NPUWorker.__init__()` |
| 3 | Patch DeepSeek-V3.2 indexer cache spec，新增独立 `IndexerKVSpec` | `vllm_ascend/patch/platform/patch_kv_cache_interface.py:152 IndexerKVSpec`；`vllm_ascend/patch/dsa_sparse/patch_deepseek_v2.py:16 _dsa_indexer_get_kv_cache_spec()` | `vllm_ascend/patch/dsa_sparse/patch_deepseek_v2.py:32 patch_deepseek_v2_indexer_cache_spec()` |
| 4 | 将 HBM KV cache 拆分为 Indexer dense plane 和 MLA resident plane | `vllm_ascend/worker/model_runner_v1.py:4307 get_kv_cache_spec()` 中 DSA MLA spec 分支；`vllm_ascend/patch/dsa_sparse/patch_kv_cache_utils.py:487 _get_kv_cache_configs()` | `vllm_ascend/dsa_sparse/dsa_spec_utils.py:30 is_dsa_indexer_spec()`；`:37 is_dsa_mla_resident_spec()` |
| 5 | 新增按 KV group 分配的 `MultiBlockPool` | `vllm_ascend/patch/dsa_sparse/patch_kv_cache_decoupling.py:47 MultiBlockPool` | 同文件 `:342 _coordinator_init()`；`:392 _get_num_blocks_to_allocate_by_group()`；`:487 _allocate_slots()` |
| 6 | 新增 DSA KV 容量计算、比例分配和容量报告 | `vllm_ascend/patch/dsa_sparse/patch_kv_cache_utils.py:106 _get_dsa_base_and_group_num_blocks()`；`:198 _get_dsa_capacity_metrics()`；`:255 _report_dsa_kv_cache_config()` | 同文件 `:463 _fix_dsa_group_num_blocks()`；`:487 _get_kv_cache_configs()` |
| 7 | 扩展 Request、SchedulerOutput 和 worker cached request metadata | `vllm_ascend/patch/dsa_sparse/patch_request.py:16`；`vllm_ascend/patch/dsa_sparse/patch_scheduler_output.py:16 NewRequestData`、`:42 CachedRequestData`、`:72 SchedulerOutput` | `vllm_ascend/patch/dsa_sparse/patch_scheduler.py:252 _populate_dsa_scheduler_output()`；`vllm_ascend/dsa_sparse/dsa_model_runner_state.py:106 update_states()` |
| 8 | 新增 `ReqStage` 请求状态机 | `vllm_ascend/dsa_sparse/dsa_types.py:42 ReqStage` | `vllm_ascend/dsa_sparse/dsa_sparse.py:1052 plan_decode_resident_slots()`；`vllm_ascend/dsa_sparse/dsa_req_meta.py:53 _build_req_forward_plan()` |
| 9 | 包装 scheduler slot allocation，实现 fixed-budget resident 分配 | `vllm_ascend/patch/dsa_sparse/patch_scheduler.py:95 _estimate_dsa_resident_slots()`；`:178 _install_dsa_allocate_slots_wrapper()` | `vllm_ascend/dsa_sparse/dsa_sparse.py:875 dsa_alloc_slots_wrap()`；`:1052 plan_decode_resident_slots()` |
| 10 | 新增 prefill/decode phase barrier，保证 scheduler/worker 状态对齐 | `vllm_ascend/patch/dsa_sparse/patch_scheduler.py:329 _dsa_sparse_schedule()`，其中 `:337 dsa_phase_barrier_active` | 同文件 `:210 _withhold_decode_running_for_prefill()`；`:239 _withhold_waiting_for_decode()`；`tests/ut/core/test_dsa_phase_barrier.py:1` |
| 11 | 新增 `DSASparseV1`，统一 scheduler 和 worker 两侧 DSA 逻辑 | `vllm_ascend/dsa_sparse/dsa_sparse.py:142 DSASparseV1` | `vllm_ascend/patch/dsa_sparse/patch_scheduler.py:312 _dsa_sparse_scheduler_init()`；`vllm_ascend/worker/worker.py:332 init_device()` 中 worker manager 创建 |
| 12 | 新增 request、forward、layer 三层 metadata 模型 | `vllm_ascend/dsa_sparse/dsa_req_meta.py:41 ReqForwardPlan`、`:147 ReqMeta`；`vllm_ascend/dsa_sparse/dsa_forward_batch.py:34 DSAModelForwardMeta`、`:100 DSAForwardSparseDecodeBatch`、`:226 DSALayerRuntimeBatch`、`:251 DSAForwardLayerBatch` | `vllm_ascend/dsa_sparse/dsa_forward_batch.py:339 _build_forward_batches_from_dsa_meta()` |
| 13 | 新增 worker-local DRAM hot KV store 和 host arena | `vllm_ascend/dsa_sparse/dsa_hot_kv_store_core.py:42 DSAHotKVStore`；`vllm_ascend/dsa_sparse/dsa_ascend_hot_kv_store.py:30 AscendDSAHotKVStore` | `vllm_ascend/dsa_sparse/dsa_hot_kv_store_core.py:108 preallocate_layer_cache()`、`:485 dump_layer_blocks_for_requests()`；`vllm_ascend/dsa_sparse/dsa_sparse.py:724 _dump_layer_full_blocks_to_dram_batch()` |
| 14 | 新增 HBM resident request pool 和固定 slot status pool | `vllm_ascend/dsa_sparse/dsa_resident_pool.py:30 DSAResidentTokenPool` | 同文件 `:82 acquire()`、`:95 release()`、`:133 get_resident_slot_token_status()`；`vllm_ascend/dsa_sparse/dsa_sparse.py:173` 初始化 resident pool |
| 15 | ModelRunner 增加 DSA forward 前后处理和请求结束/抢占清理 | `vllm_ascend/worker/model_runner_v1.py:1456 prepare_dsa_scheduled_request()`；`:1493 post_process_dsa_after_model_forward()` | `vllm_ascend/dsa_sparse/dsa_model_runner_state.py:41 dsa_request_finished_in_worker()`、`:47 dsa_request_preempted_in_worker()`；`vllm_ascend/worker/model_runner_v1.py:2160` 调用入口 |
| 16 | MLA/SFA 增加 attention begin、after indexer、finished hook | `vllm_ascend/ops/mla.py:195 mla_forward()`，其中 `:208` 安装 layer hook；`vllm_ascend/dsa_sparse/dsa_sparse.py:706 attention_begin()`、`:776 attention_finished()`、`:797 after_indexer()` | `vllm_ascend/attention/sfa_v1.py:1910 maybe_prepare_dsa_indexer_score_controls()`；`:1920 maybe_execute_after_dsa_indexer()` |
| 17 | lightning-indexer 改为 full-batch mixed key length 路径 | `vllm_ascend/attention/sfa_v1.py:237 _run_dsa_full_batch_lightning_indexer()`；`:1386 build_dsa_mixed_key_lens()`；`:1425` full-batch 调用 | `vllm_ascend/dsa_sparse/dsa_batch_tensor_utils.py:77 build_dsa_mixed_key_lens()`；`tests/ut/attention/test_dsa_key_lens.py:1` |
| 18 | 新增 row-mode GS 自定义算子及 Torch binding/meta 注册 | `csrc/gather_selection_kv_cache/op_kernel/gather_selection_kv_cache.cpp:28 gather_selection_kv_cache()`；`csrc/torch_binding.cpp:1211` op schema | `vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py:221 gather_selection_update()`、`:382` 算子调用；`csrc/torch_binding_meta.cpp:247 gather_selection_kv_cache_meta()` |
| 19 | GS 支持 PAD/DENSE/SPARSE、不同序列长度和不同 decode 轮次混合 | `vllm_ascend/dsa_sparse/dsa_types.py:26 DSADecodeRowMode`；`vllm_ascend/dsa_sparse/dsa_forward_batch.py:497` row mode 构造 | `csrc/gather_selection_kv_cache/op_kernel/gather_selection_kv_cache_split_bs_reuse_vec.h:36` row mode 常量及 `:107 InitRowMode()`；`tests/ut/dsa_sparse/test_dsa_forward_row_mode.py:1` |
| 20 | 新增 SFA resident logical indices 构造 | `vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py:352` attention indices 分配；`:370-400` GS 原址状态更新和输出 | `vllm_ascend/dsa_sparse/dsa_sparse.py:630-642` 提交本层 indices；`vllm_ascend/attention/sfa_v1.py:1479-1502` 交给 SFA；`tests/ut/dsa_sparse/test_dsa_rowmode_attention_indices.py:1` |
| 21 | 新增独立 DSA graph phase、gate 和 graph-stable buffers | `vllm_ascend/dsa_sparse/dsa_graph_gate.py:19 DSA_GRAPH_PHASE_ROW_MODE_DECODE`；`:72 evaluate_dsa_row_mode_decode_graph()`；`vllm_ascend/patch/dsa_sparse/patch_cudagraph_phase.py:44 _dispatch_with_dsa_graph_phase()` | `vllm_ascend/dsa_sparse/dsa_graph_buffers.py:71 DSAGraphBuffersMixin`；`vllm_ascend/patch/dsa_sparse/patch_cudagraph_phase.py:120 patch_cudagraph_dispatcher_phase()` |
| 22 | 新增 graph capture/replay 的 batch row 映射和 padding | `vllm_ascend/dsa_sparse/dsa_graph_buffers.py:373 prepare_row_mode_decode_graph_capture_batch()`；`:432 prepare_row_mode_decode_graph_replay_batch()` | `vllm_ascend/dsa_sparse/dsa_forward_batch.py:135 query_position_rows_tensor`、`:150 batch_row_indices_tensor`、`:156 active_batch_row_indices_tensor`；`tests/ut/patch/dsa_sparse/test_cudagraph_phase.py:1` |
| 23 | 新增 GS hit/miss、overlap、miss rate trace | `vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py:65 _build_gather_selection_overlap_stats()`；`:340` 统计入口 | `vllm_ascend/dsa_sparse/dsa_trace.py:20 DSA_TRACE_POINT_GATHER_SELECTION_STATS` |
| 24 | 新增 request/layer/rank 过滤的 trace 配置 | `vllm_ascend/dsa_sparse/dsa_trace.py:29 DSATraceConfig`；`:92 configure_dsa_trace()`；`:144 dsa_trace_enabled()` | `vllm_ascend/dsa_sparse/dsa_config.py:46 _normalize_dsa_trace_points_config()`；`vllm_ascend/worker/model_runner_v1.py:294` 初始化 trace |
| 25 | 新增 block overflow、patch alias、KV group 缺失等 fail-fast 检查 | `vllm_ascend/patch/dsa_sparse/patch_scheduler.py:121 _check_dsa_block_ids_for_overflow()`；`vllm_ascend/patch/dsa_sparse/patch_engine_process.py:62 verify_dsa_runtime_patches_installed()` | `vllm_ascend/worker/worker.py:680 get_kv_cache_spec()`，其中 `:700` 检查 IndexerKVSpec；`tests/ut/patch/dsa_sparse/test_patch_aliases.py:1` |
| 26 | 新增 demo、QA dataset 和准确率评估脚本 | `examples/dsa_demo/simple_prompt_test.py:57 main()`；`examples/dsa_demo/qa_dataset_test.py:388 main()`；`examples/dsa_demo/eval_dataset_acc_score.py:229` 主入口 | `examples/dsa_demo/readme.md` 为运行参数说明，不计入源码文件清单 |
| 27 | 新增状态机、row-mode、graph gate、phase barrier、patch alias 单元测试 | `tests/ut/dsa_sparse/test_dsa_forward_row_mode.py:1`；`tests/ut/dsa_sparse/test_dsa_graph_gate.py:1`；`tests/ut/dsa_sparse/test_dsa_rowmode_attention_indices.py:1` | `tests/ut/core/test_dsa_phase_barrier.py:1`；`tests/ut/patch/dsa_sparse/test_cudagraph_phase.py:1`；`tests/ut/patch/dsa_sparse/test_patch_aliases.py:1`；`tests/ut/attention/test_dsa_key_lens.py:1` |

## 6. 提交演进

| Commit | 主要内容 |
| --- | --- |
| `6275d200c` | 首版 fixed-budget swap，引入 GS、qk-score、index-update、H2D scatter 等原型 |
| `f581715a9` | eager continuous batching，支持 dense/sparse 和长短请求混合 |
| `84846311d` | GS 图模式初步适配；删除 qk-score、index-update、scatter-copy 等无效路径 |
| `7e7b3a396` | mixed-length decode 管控面、metadata 和 GS eager 接口统一 |
| `2f96f9e98` | mixed-length decode 图模式适配 |
| `4db7ee198` | 重构 DSA 入参、测试脚本和 trace |
| `fdd363673` | 将依赖 vLLM 侧的修改迁移到 vLLM-Ascend plugin/patch 架构 |
| `e84688929` | metadata 重命名和生命周期结构调整 |
| `c5ebdb251` | 增加 GS hit/miss 和 overlap 调测 |
| `a0e31c57e` | resident slot status 从 backend 临时资源迁移到持久 resident pool |
| `ad428d7bf` | 更新分支 README 和详细设计文档 |

## 7. 修改的源码文件

以下文件来自 `git diff --name-only da421afad7192dac64e39ae1d32305d57344f3cf..HEAD`，已经排除 README、`docs/` 和 `examples/dsa_demo/readme.md`。

### 7.1 构建和自定义算子：13 个

1. `setup.py`
2. `csrc/build_aclnn.sh`
3. `csrc/torch_binding.cpp`
4. `csrc/torch_binding_meta.cpp`
5. `csrc/gather_selection_kv_cache/gather_selection_kv_cache_torch_adpt.h`
6. `csrc/gather_selection_kv_cache/op_host/CMakeLists.txt`
7. `csrc/gather_selection_kv_cache/op_host/gather_selection_kv_cache_def.cpp`
8. `csrc/gather_selection_kv_cache/op_host/gather_selection_kv_cache_proto.cpp`
9. `csrc/gather_selection_kv_cache/op_host/gather_selection_kv_cache_tiling.cpp`
10. `csrc/gather_selection_kv_cache/op_host/gather_selection_kv_cache_tiling.h`
11. `csrc/gather_selection_kv_cache/op_kernel/gather_selection_kv_cache.cpp`
12. `csrc/gather_selection_kv_cache/op_kernel/gather_selection_kv_cache_split_bs_reuse.h`
13. `csrc/gather_selection_kv_cache/op_kernel/gather_selection_kv_cache_split_bs_reuse_vec.h`

### 7.2 vLLM-Ascend 核心运行时：27 个

1. `vllm_ascend/__init__.py`
2. `vllm_ascend/platform.py`
3. `vllm_ascend/utils.py`
4. `vllm_ascend/attention/sfa_v1.py`
5. `vllm_ascend/attention/utils.py`
6. `vllm_ascend/ops/mla.py`
7. `vllm_ascend/worker/block_table.py`
8. `vllm_ascend/worker/model_runner_v1.py`
9. `vllm_ascend/worker/worker.py`
10. `vllm_ascend/dsa_sparse/__init__.py`
11. `vllm_ascend/dsa_sparse/dsa_ascend_hot_kv_store.py`
12. `vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py`
13. `vllm_ascend/dsa_sparse/dsa_attention_layout.py`
14. `vllm_ascend/dsa_sparse/dsa_batch_tensor_utils.py`
15. `vllm_ascend/dsa_sparse/dsa_config.py`
16. `vllm_ascend/dsa_sparse/dsa_forward_batch.py`
17. `vllm_ascend/dsa_sparse/dsa_graph_buffers.py`
18. `vllm_ascend/dsa_sparse/dsa_graph_gate.py`
19. `vllm_ascend/dsa_sparse/dsa_hot_kv_store_core.py`
20. `vllm_ascend/dsa_sparse/dsa_layer_cache_zones.py`
21. `vllm_ascend/dsa_sparse/dsa_model_runner_state.py`
22. `vllm_ascend/dsa_sparse/dsa_req_meta.py`
23. `vllm_ascend/dsa_sparse/dsa_resident_pool.py`
24. `vllm_ascend/dsa_sparse/dsa_sparse.py`
25. `vllm_ascend/dsa_sparse/dsa_spec_utils.py`
26. `vllm_ascend/dsa_sparse/dsa_trace.py`
27. `vllm_ascend/dsa_sparse/dsa_types.py`

### 7.3 Runtime Patch：17 个

1. `vllm_ascend/patch/__init__.py`
2. `vllm_ascend/patch/dsa_sparse/__init__.py`
3. `vllm_ascend/patch/dsa_sparse/patch_cudagraph_phase.py`
4. `vllm_ascend/patch/dsa_sparse/patch_deepseek_v2.py`
5. `vllm_ascend/patch/dsa_sparse/patch_engine_core.py`
6. `vllm_ascend/patch/dsa_sparse/patch_engine_process.py`
7. `vllm_ascend/patch/dsa_sparse/patch_kv_cache_decoupling.py`
8. `vllm_ascend/patch/dsa_sparse/patch_kv_cache_utils.py`
9. `vllm_ascend/patch/dsa_sparse/patch_request.py`
10. `vllm_ascend/patch/dsa_sparse/patch_runtime.py`
11. `vllm_ascend/patch/dsa_sparse/patch_scheduler.py`
12. `vllm_ascend/patch/dsa_sparse/patch_scheduler_output.py`
13. `vllm_ascend/patch/platform/__init__.py`
14. `vllm_ascend/patch/platform/patch_dsa_sparse.py`
15. `vllm_ascend/patch/platform/patch_kv_cache_interface.py`
16. `vllm_ascend/patch/worker/__init__.py`
17. `vllm_ascend/patch/worker/patch_dsa_sparse.py`

### 7.4 示例源码：3 个

1. `examples/dsa_demo/eval_dataset_acc_score.py`
2. `examples/dsa_demo/qa_dataset_test.py`
3. `examples/dsa_demo/simple_prompt_test.py`

### 7.5 单元测试源码：7 个

1. `tests/ut/attention/test_dsa_key_lens.py`
2. `tests/ut/core/test_dsa_phase_barrier.py`
3. `tests/ut/dsa_sparse/test_dsa_forward_row_mode.py`
4. `tests/ut/dsa_sparse/test_dsa_graph_gate.py`
5. `tests/ut/dsa_sparse/test_dsa_rowmode_attention_indices.py`
6. `tests/ut/patch/dsa_sparse/test_cudagraph_phase.py`
7. `tests/ut/patch/dsa_sparse/test_patch_aliases.py`

### 7.6 数量校验

```text
构建和自定义算子  13
核心运行时        27
Runtime Patch     17
示例源码           3
单元测试源码       7
--------------------
合计              67
```

## 8. 当前支持范围和限制

1. 当前仅明确适配 `DeepseekV32ForCausalLM`。
2. 依赖 `async_scheduling=False`，scheduler output 必须和 worker forward 状态严格对齐。
3. prefix cache、spec decode、chunked prefill、prefill/decode mixed batch 暂未支持。
4. PCP sparse path 暂不支持。
5. Ascend balance、dynamic、recompute 和 profiling scheduler 与 DSA 不兼容。
6. 图模式只覆盖稳态 single-token decode；转换阶段和 full block dump 仍走 eager。
7. GS 性能优化仍是 TODO，miss rate 可能显著影响算子延迟。
8. GS trace 可能触发 D2H 和设备同步，只能用于调试。
9. 开启 DSA 后如果没有打印 Indexer/MLA 两组 KV cache 的容量报告，说明 split cache 或 patch 很可能没有生效。
10. 当前提交中仍存在 `[WIP]` 和后续优化项，属于原型分支而非完全收敛的生产实现。

## 9. 代码质量观察

- `git diff --check` 在 GS 自定义算子和设计文档中发现若干 trailing whitespace，不影响运行逻辑，但建议在合入前清理。
- 本文为基于提交历史、最终 diff 和当前源码调用链的静态分析，未执行依赖 Ascend NPU 的端到端推理测试。
- 分支已经补充多个 CPU 可执行的 metadata、状态机、graph gate 和 patch alias 单元测试，但 NPU 自定义算子和端到端图 replay 仍需在实际设备验证。
