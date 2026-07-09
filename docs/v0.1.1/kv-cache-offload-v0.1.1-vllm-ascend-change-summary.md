# v0.1.1 vllm-ascend 变更摘要

> 状态：Summary
> 来源仓库：`/home/solidyang/workspace/vllm-ascend`
> 比较范围：`36e15a2fdc..HEAD`
> 排除范围：`tests/**`
> 目标分支：`feat/kv-offload-v011-compact-sfa`

本文汇总 `36e15a2fdc` 之后 vllm-ascend 中与 KV cache offload v0.1.1 相关的非测试代码变更，便于和 ASU-Ascend v0.1.1 设计、运行指南对照。

当前非测试变更共 29 个文件，约 `2613 insertions, 2 deletions`。测试文件未纳入本文清单。

## 1. 提交主题

`36e15a2fdc` 之后的提交主题如下：

| 提交 | 主题 |
|---|---|
| `6e867c102e` | Add v0 KV offload validation path |
| `7a139ee59a` | feat: wire asu hbm index real ops |
| `a374ca2a95` | feat: add kv offload compact sfa path |
| `38beaf1d40` | feat: carve offload pinned blocks out of the normal KV allocator |
| `db009081c2` | feat: add pure-Python reference HBM index ops for bring-up |
| `0649e5986a` | feat: log KV offload persist/validate/compact stats for layer 0 |
| `eebfd969bd` | feat: trace kv offload hbm index ops |
| `29915086f7` | feat: add direct aicpu maintain path |
| `df323df8a1` | feat: add direct lookup op path |
| `7fa57d402b` | feat: materialize generated tokens in compact offload |
| `682dd59479` | fix: preserve invalid compact SFA topk sentinel |

## 2. 总体变化

这批改动把 KV offload 从 v0.1 的旁路校验推进到 v0.1.1 compact SFA 调试路径：

1. prefill 阶段把 MLA KV token 序列化后写入 MicroKV。
2. decode 阶段使用 HBM index lookup 将原始 `token_pos` 映射到 resident `slot_id`。
3. lookup miss 时从 MicroKV 取回 KV，写入旁路 cache 或 vLLM 预留的 compact offload blocks。
4. maintain 阶段保护本轮 query 命中的 slot，回收未保护 slot 到 free pool。
5. compact SFA 路径把原始 `topk_indices(token_pos)` 转换为 `topk_indices(slot_id)`，并替换为 compact block table，使 SFA 读取 vLLM KV tensor 尾部预留的 offload blocks。
6. model runner 和 worker 负责 eager 模式校验、环境变量开关、MicroKV client 初始化、direct `.so` / reference op 注入，以及从普通 KV allocator 中 carve out pinned offload blocks。

当前路径主要用于 eager bring-up。它不覆盖图捕获、DSA CP、Sparse C8 indexer，以及非 `DecodeOnly` 的 compact SFA 执行。

## 3. Python / 运行时文件

| 文件 | 状态 | 功能 |
|---|---|---|
| `vllm_ascend/envs.py` | M | 新增 KV offload 相关环境变量，包括 validate、compact SFA、最大 pinned requests、reference HBM ops、direct lookup / maintain `.so`、trace index ops、MicroKV socket 等。 |
| `vllm_ascend/attention/offload_kv_cache_v0.py` | A | 核心 offload manager。负责 MLA KV token record 的 pack / unpack、prefill 写 MicroKV、HBM index 状态维护、lookup / maintain 调用、v0.1 validate 路径和 v0.1.1 compact SFA 路径。 |
| `vllm_ascend/attention/offload_kv_cache_v0_ownership.py` | A | compact SFA 的 block ownership 工具。计算每 request 需要的 compact blocks、offload 预留 blocks、预留内存大小、膨胀 KV tensor size，并校验 normal KV block 与 offload block 不混用。 |
| `vllm_ascend/attention/offload_kv_cache_v0_ref_ops.py` | A | 纯 Python HBM index lookup / maintain 参考实现。用于真实新算子未注册时的 bring-up，也方便在 host 侧验证 lookup / eviction 语义。 |
| `vllm_ascend/attention/sfa_v1.py` | M | 在 SFA forward 中接入 KV offload：prefill 后持久化 KV；decode 时执行 validate 或 compact SFA 输入改写；对 layer 0 输出 persist / validate / compact 统计日志；compact 路径保留 invalid topk sentinel。 |
| `vllm_ascend/attention/utils.py` | M | `AscendCommonAttentionMetadata` 增加 CPU 侧 request / token 映射字段：`req_ids`、`token_req_indices_cpu`、`token_positions_cpu`、`prefill_lens_cpu`，供 offload eager SFA 路径定位原始 token。 |
| `vllm_ascend/ascend_forward_context.py` | M | forward context 新增 `offload_kv_cache_v0`，让 attention 层可从 `_EXTRA_CTX` 获取 model runner 创建的 offload manager。 |
| `vllm_ascend/worker/model_runner_v1.py` | M | 初始化 `OffloadKVCacheV0Manager`；根据环境变量选择真实 torch op、Python reference op 或 direct ASU `.so`；收集 CPU 侧 request / token metadata；request 结束时释放 offload 状态；compact SFA 时预留并注册 offload block pool。 |
| `vllm_ascend/worker/worker.py` | M | 在 profile 可用 KV cache 内存时，为 compact SFA offload pinned block pool 先扣除内存预算，保证 engine 计算出的 normal KV block 数不会覆盖预留 offload blocks。 |

## 4. Python 调用流程深拆

本节按运行时的真实调用方向，从 worker / model runner 入口一路拆到 SFA attention 内部。核心分两条路径：

1. `VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE=1`：v0.1 旁路校验路径。SFA 仍读原始 KV cache，只额外验证 MicroKV + HBM index 维护出来的旁路 KV 是否和原始 KV 一致。
2. `VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1`：v0.1.1 compact SFA 路径。SFA 的 `topk_indices`、`block_table`、`actual_seq_lengths_kv` 被改写到 compact slot 坐标系，让 SFA 实际读取 vLLM KV tensor 尾部预留的 offload blocks。

### 4.1 启动与 manager 初始化

入口在 `NPUModelRunner.__init__()`：

```text
NPUModelRunner.__init__
  -> 读取 envs.VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE / COMPACT_SFA
  -> 校验 eager 模式
  -> compact SFA 时校验 MAX_PINNED_REQS > 0
  -> 创建 MicroKV KVStoreClient
  -> 选择 lookup / maintain op 来源
  -> 创建 OffloadKVCacheV0Manager
```

op 选择顺序如下：

| 条件 | lookup / maintain 来源 |
|---|---|
| `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` | `offload_kv_cache_v0_ref_ops.py` 的纯 Python reference op |
| direct `.so` env 已设置 | `direct_lookup.py` / `direct_maintain.py` 用 `ctypes` 加载 ASU direct shared library |
| 以上都未设置 | `OffloadKVCacheV0Manager._call_lookup()` / `_call_maintain()` 内部回退到 `torch.ops._C_ascend.asu_hbm_index_*` |

最终创建的 manager 持有：

| 字段 | 作用 |
|---|---|
| `client=KVStoreClient(envs.MICROKV_SOCKET)` | MicroKV 读写入口 |
| `compact_sfa_enabled` | 决定走 validate 还是 compact SFA |
| `max_pinned_reqs` / `block_size` | 计算 compact offload block pool 大小 |
| `lookup_op` / `maintain_op` | 可注入的 lookup / maintain 实现 |
| `trace_index_ops` | 控制 lookup / maintain free-slot 统计日志 |

### 4.2 compact SFA 的内存 carve-out

compact SFA 需要一段不会被普通 scheduler 分配的 KV 物理 blocks。这个隔离分两步完成。

第一步发生在 `NPUWorker.determine_available_memory()`：

```text
NPUWorker.determine_available_memory
  -> profile 得到 available_kv_cache_memory_bytes
  -> _reserve_offload_kv_cache_memory
       -> 如果未启用 compact SFA：原样返回
       -> get_kv_cache_spec()
       -> reserved_blocks = offload_manager.offload_reserved_blocks()
       -> reserved_bytes = reserved_blocks * sum(attention page_size_bytes)
       -> 返回 available - reserved_bytes
```

这一步影响 engine 侧计算出的 `kv_cache_config.num_blocks`。也就是说，scheduler 可见的 normal KV block 数已经扣除了 offload pool 对应的内存。

第二步发生在 `NPUModelRunner.initialize_kv_cache()`：

```text
NPUModelRunner.initialize_kv_cache
  -> deepcopy(kv_cache_config)
  -> compact SFA 时 _reserve_offload_blocks_in_kv_cache_config
       -> 对 attention KV tensor 的 size 增加 reserved_blocks * page_size_bytes
  -> initialize_kv_cache_tensors(kv_cache_config)
  -> 校验 total_tensor_blocks == kv_cache_config.num_blocks + reserved_blocks
  -> offload_manager.register_static_offload_block_pool(total_blocks)
```

这一步把刚才扣掉的内存以 tensor 尾部 blocks 的形式加回来。最终关系是：

```text
scheduler-visible normal blocks: [0, kv_cache_config.num_blocks)
offload pinned blocks:           [kv_cache_config.num_blocks, total_tensor_blocks)
```

`register_static_offload_block_pool()` 会构造 `BlockOwnershipRegistry`，并把尾部 blocks 切成每个 request 一行的 compact block table row。后续 compact SFA 只通过这些 offload rows 读取 KV。

### 4.3 每个 batch 的 CPU metadata 传递

offload 路径需要知道每个 token 属于哪个 request、它在原始序列中的 position、以及该 request 的 prefill 长度。这些信息在 `NPUModelRunner._prepare_inputs()` 中落到 CPU tensor：

```text
NPUModelRunner.execute_model
  -> _update_states(scheduler_output)
  -> 对 finished_req_ids 调 offload_manager.release_request(req_id)
  -> _prepare_inputs(...)
       -> self.offload_kv_cache_v0_req_ids
       -> self.offload_kv_cache_v0_token_req_indices_cpu
       -> self.offload_kv_cache_v0_token_positions_cpu
       -> self.offload_kv_cache_v0_prefill_lens_cpu
```

随后 `NPUModelRunner._build_attention_metadata()` 把这些字段挂到 `AscendCommonAttentionMetadata`：

```text
AscendCommonAttentionMetadata(
  req_ids=...,
  token_req_indices_cpu=...,
  token_positions_cpu=...,
  prefill_lens_cpu=...,
)
```

SFA metadata builder 再把它们复制到 `AscendSFAMetadata`。因此，进入 `AscendSFAImpl.forward()` 时，attention 层可以通过 `attn_metadata` 反查：

| 字段 | 用途 |
|---|---|
| `req_ids[req_index]` | 生成 MicroKV key 和 per-request cache key |
| `token_req_indices_cpu[token_index]` | 从 token index 找到 request index |
| `token_positions_cpu[token_index]` | 找到原始 `token_pos`，作为 MicroKV key 和 HBM index query |
| `prefill_lens_cpu[req_index]` | 区分 prefill token 和 decode 过程中生成的新 token |

### 4.4 forward context 注入

`execute_model()` 在真正调用模型 forward 前会包一层 `set_ascend_forward_context()`：

```text
NPUModelRunner.execute_model
  -> set_ascend_forward_context(
       attn_metadata,
       ...,
       offload_kv_cache_v0=self.offload_kv_cache_v0,
     )
       -> forward_context.offload_kv_cache_v0 = offload_kv_cache_v0
```

`ascend_forward_context.py` 同时把 `offload_kv_cache_v0` 加进 `_ExtraForwardContextProxy` 允许访问的 extra attrs。这样 `AscendSFAImpl.forward()` 内部可以通过：

```python
offload_kv_cache_v0 = _EXTRA_CTX.offload_kv_cache_v0
```

拿到同一个 manager 实例。

### 4.5 SFA forward 内的总控顺序

`AscendSFAImpl.forward()` 中 KV offload 的调用点在 SFA 真正计算前，顺序是：

```text
AscendSFAImpl.forward
  -> 从 _EXTRA_CTX 取 offload_kv_cache_v0
  -> 启用时做限制校验
       -> 不支持 DSA CP
       -> 不支持 Sparse C8 indexer
       -> 不支持 graph capture
  -> persist_prefill_kv_to_microkv(...)
  -> indexer_select_post_process(...) 生成原始 topk_indices(token_pos)
  -> if compact_sfa_enabled and DecodeOnly:
       prepare_compact_sfa_inputs(...)
       用 compact topk / compact block_table / compact seq len 替换 SFA 输入
     elif compact_sfa_enabled and 有 decode token 但不是 DecodeOnly:
       raise ValueError
     else:
       validate_topk_with_real_hbm_index_ops(...)
       SFA 继续使用原始 topk / block_table
  -> _execute_sparse_flash_attention_process(...)
```

layer 0 会额外打印 persist、validate 或 compact 统计，避免每层刷屏。

### 4.6 prefill 阶段：写 MicroKV 并初始化 resident slots

`persist_prefill_kv_to_microkv()` 是所有路径共用的 prefill 持久化入口：

```text
persist_prefill_kv_to_microkv(layer_name, kv_cache, slot_mapping, attn_metadata)
  -> parse_layer_id(layer_name)
  -> flatten kv_cache[0] / kv_cache[1]
  -> 如果 attn_state 是 DecodeOnly 或 SpecDecoding：跳过
  -> 遍历 num_actual_tokens
       -> req_index = token_req_indices_cpu[token_index]
       -> token_pos = token_positions_cpu[token_index]
       -> prefill_len = prefill_lens_cpu[req_index]
       -> original_slot = slot_mapping[token_index]
       -> token_pos >= prefill_len 或 original_slot < 0 时跳过
       -> key = make_microkv_mla_token_key(req_id, layer_id, token_pos)
       -> value = pack_mla_token_record(k_nope_flat[slot], k_pe_flat[slot])
  -> client.batch_put(cache_type, keys, values)
  -> _write_resident_prefill_token(...)
```

`pack_mla_token_record()` 会写入 record magic、version、dtype、shape、payload 长度和 checksum。这样 decode 阶段从 MicroKV 读回时能做 dtype / shape / checksum 校验。

`_write_resident_prefill_token()` 的作用是把 prefill 的前 `resident_slot_count` 个 token 直接建立 resident 映射：

| 路径 | resident KV 存放位置 |
|---|---|
| validate 路径 | manager 自己分配的旁路 `state.k_nope_cache` / `state.k_pe_cache` |
| compact SFA 路径 | vLLM KV tensor 尾部预留的 offload physical blocks |

同时它会更新：

```text
state.index[token_pos] = slot_id
state.slot_to_index[slot_id] = token_pos
```

因此，后续 lookup 命中 resident token 时不需要再从 MicroKV 加载。

### 4.7 v0.1 validate 路径：旁路校验，不改 SFA 输入

当没有启用 compact SFA 时，SFA 计算仍使用原始 `topk_indices` 和原始 `block_table`。offload manager 只做旁路一致性校验：

```text
validate_topk_with_real_hbm_index_ops(layer_name, kv_cache, topk_indices, attn_metadata)
  -> parse_layer_id(layer_name)
  -> 只处理 decode tokens；没有 decode tokens 直接返回
  -> 按 req_index 分组 decode token
  -> 对每个 req:
       -> 收集 topk 中合法的 prefill token_pos
       -> get_or_create_bypass_cache(req_id, layer_id, ...)
       -> _prepare_query_index(state, valid_query_tokens)
       -> _call_lookup(state, query_index)
       -> _load_query_tokens_to_bypass_cache(...)
       -> 对每个 topk token:
            -> 用原始 block_table + token_pos 算 original_slot
            -> 比较 original KV 和 bypass KV
            -> mismatch 时 strict 模式 raise
       -> state.last_query_slots = slot_out
       -> free_head > 0 时 _call_maintain(state)
```

这条路径验证的是：

1. MicroKV 中存的 KV record 能按 key 读回。
2. HBM index lookup 能把 `token_pos` 映射到 slot。
3. miss 后加载到旁路 cache 的 KV 与原始 vLLM KV cache 一致。
4. maintain 能按 `last_query_slots` 保护本轮热 slot 并回收其余 slot。

它不会改变传给 `npu_sparse_flash_attention` 的输入。

### 4.8 v0.1.1 compact SFA 路径：改写 SFA 输入

compact SFA 只支持 `DecodeOnly`。进入 `prepare_compact_sfa_inputs()` 后，流程是：

```text
prepare_compact_sfa_inputs(layer_name, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_kv)
  -> 校验 compact_sfa_enabled
  -> 确保已注册 offload block pool
  -> _assert_original_kv_metadata(attn_metadata)
       -> 原始 block_table / slot_mapping 只能引用 normal KV blocks
  -> 按 req_index 分组 decode token
  -> 对每个 req:
       -> _get_or_allocate_offload_block_row(req_id)
       -> compact_actual_seq_lengths[req_index] = compact_blocks_per_req * block_size
       -> 计算 key_len
       -> _collect_compact_query_tokens(...)
       -> get_or_create_hbm_index_state(req_id, layer_id, device)
       -> _prepare_query_index(...)
       -> _call_lookup(...)
       -> _load_query_tokens_to_compact_cache(...)
       -> 把 compact_topk_indices 中的 token_pos 改写为 slot_id
       -> state.last_query_slots = slot_out
       -> free_head > 0 时 _call_maintain(state)
  -> 返回 CompactSFAInputs(topk_indices, block_table, actual_seq_lengths_kv)
```

这里有三个关键转换。

第一，原始 token 坐标转换为 compact slot 坐标：

```text
topk_indices: token_pos -> slot_id
```

负数 sentinel 保持为负数，不参与 lookup，也不会被改写。这对应最后一个 fix 提交。

第二，KV 数据被 materialize 到 compact offload blocks：

| token 类型 | 数据来源 | 写入位置 |
|---|---|---|
| `token_pos < prefill_len` | MicroKV record | `physical_slot_for_compact_slot(slot_id, block_size, offload_block_row)` |
| `token_pos >= prefill_len` | 原始 vLLM KV cache 中的 generated token | 同一个 compact physical slot |

这就是 `feat: materialize generated tokens in compact offload` 的核心意义：decode 过程中产生的新 token 不在 MicroKV prefill backing store 内，需要从原始 KV cache 复制到 compact offload cache，否则 SFA 读 compact 坐标时会缺数据。

第三，SFA metadata 被替换为 compact block table：

```text
sfa_topk_indices = compact_sfa_inputs.topk_indices
sfa_attn_metadata = replace(attn_metadata, block_table=compact_sfa_inputs.block_table)
sfa_actual_seq_lengths_key = compact_sfa_inputs.actual_seq_lengths_kv
```

最终 `npu_sparse_flash_attention` 仍接收原始 `kv_cache` tensor，但它通过 compact `block_table` 只寻址到 tensor 尾部的 offload physical blocks。

### 4.9 lookup / maintain 的实际分发

manager 内部统一通过 `_call_lookup()` / `_call_maintain()` 调用 HBM index 算子：

```text
_call_lookup
  -> 如果 self.lookup_op 不为 None：调用注入实现
  -> 否则 torch.ops._C_ascend.asu_hbm_index_lookup(...)

_call_maintain
  -> 如果 self.maintain_op 不为 None：调用注入实现
  -> 否则 torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(...)
```

三类实现共享同一签名：

| 实现 | 特点 |
|---|---|
| 真实 torch op | 走 vllm-ascend `_C_ascend` 注册算子 |
| direct ASU `.so` | `ctypes` 传当前 NPU stream 和 tensor data ptr，保留 eager 调试能力 |
| Python reference op | 先把 tensor 搬到 CPU list 模拟，再 copy 回 tensor；只用于语义 bring-up |

lookup 会更新 `index`、`slot_to_index`、`free_head` 并返回 `slot_out`。maintain 会根据 `last_query_slots` 保护热 slot，回收未保护 slot，并把 `free_head` 拉回可用状态。

### 4.10 request 生命周期与状态释放

offload manager 的状态主要按 `(req_id, layer_id)` 缓存：

| 状态 | 作用 |
|---|---|
| `HBMIndexLayerState.index` | `token_pos -> slot_id` |
| `HBMIndexLayerState.slot_to_index` | `slot_id -> token_pos` |
| `free_slots` / `free_head` | miss 分配和 maintain 回收的 free pool |
| `last_query_slots` | 本轮 lookup 结果，maintain 用它保护热 slot |
| `_req_offload_block_rows` | compact SFA 中每个 request 独占的 offload block row |

每次 `execute_model()` 先调用 `_update_states(scheduler_output)`，随后对 `scheduler_output.finished_req_ids` 调：

```text
offload_manager.release_request(req_id)
```

释放逻辑包括：

1. 把该 request 的 offload block row 放回 `_free_offload_block_rows`。
2. 删除所有 `(req_id, layer_id)` 的 HBM index state。
3. 清除 disabled cache 标记。

因此，compact offload block row 的生命周期跟 request 生命周期绑定，而不是跟单个 layer 或单个 decode step 绑定。

### 4.11 关键限制与不变量

| 不变量 | 说明 |
|---|---|
| eager only | manager 初始化和 SFA forward 中都会限制图捕获路径。 |
| compact SFA only DecodeOnly | 混合 prefill/decode 或其他 attention state 会 fail-fast。 |
| normal/offload block 隔离 | 原始 `block_table` / `slot_mapping` 必须只引用 normal KV blocks；compact table 必须只引用 offload KV blocks。 |
| per request compact row | 一个 request 对应一行 offload blocks，跨 layer 复用该 row，但 HBM index state 仍按 `(req_id, layer_id)` 区分。 |
| query count 固定上限 | 当前 manager 默认 `QUERY_COUNT = 2048`，超过会 raise。 |
| slot/index 固定容量 | 当前默认 `INDEX_SIZE = 128K`、`SLOT_COUNT = 10K`、resident `8K`、free `2K`；`VLLM_ASCEND_KV_OFFLOAD_V0_CAPACITY` 已标注 deprecated，compact 构造未使用它改容量。 |
| MicroKV key 维度 | key 包含 request hash、layer id、token pos、cache type，因此同 request 不同 layer / token 不冲突。 |
| invalid topk sentinel | compact rewrite 跳过负数 topk，保持 sentinel 原值。 |

## 5. C++ / Ascend 算子文件

| 文件 | 状态 | 功能 |
|---|---|---|
| `csrc/torch_binding.cpp` | M | 注册两个新 torch op：`asu_hbm_index_lookup` 和 `asu_hbm_index_maintain_aicpu`，供 Python offload manager 默认调用。 |

### 5.1 `asu_hbm_index_lookup`

| 文件 | 状态 | 功能 |
|---|---|---|
| `csrc/asu_hbm_index_lookup/README.md` | A | lookup 算子 bring-up 说明，记录当前可运行路径是 ASU direct lookup `.so`，vllm-ascend OPP 源码保留为未来 packaged custom op 路径。 |
| `csrc/asu_hbm_index_lookup/asu_hbm_index_lookup_torch_adpt.h` | A | PyTorch adapter。校验输入均为 int32，创建 `slot_out`，调用 `aclnnAsuHbmIndexLookup`。 |
| `csrc/asu_hbm_index_lookup/op_host/CMakeLists.txt` | A | lookup op 的 host、tiling、proto 编译接入配置。 |
| `csrc/asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_def.cpp` | A | lookup op 定义。声明 `index`、`slot_to_index`、`free_slots`、`free_head`、`query_index` 输入，`slot_out` 输出，以及 `req_num` attr 和 AICore 配置。 |
| `csrc/asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_proto.cpp` | A | lookup op shape / dtype 推导。`slot_out` shape 跟 `query_index` 一致，dtype 为 int32。 |
| `csrc/asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_tiling.cpp` | A | lookup tiling。读取 `req_num`，根据 AIV core 数设置 block dim，并写入 tiling data。 |
| `csrc/asu_hbm_index_lookup/op_host/asu_hbm_index_lookup_tiling.h` | A | lookup tiling data 定义，仅保存 `reqNum`。 |
| `csrc/asu_hbm_index_lookup/op_kernel/asu_hbm_index_lookup.cpp` | A | AICore lookup kernel。对每个 query token 查 `index`；miss 时从 `free_slots[free_head]` 分配 slot，更新 `index`、`slot_to_index` 和 `free_head`，输出 `slot_out`。 |
| `csrc/asu_hbm_index_lookup/tmp/README.md` | A | direct lookup debug wrapper 的构建方式和 C ABI 说明。 |
| `csrc/asu_hbm_index_lookup/tmp/direct_lookup.py` | A | 通过 `ctypes` 加载 ASU lookup `.so`，封装为与 `torch.ops._C_ascend.asu_hbm_index_lookup` 同签名的 Python callable。 |

### 5.2 `asu_hbm_index_maintain_aicpu`

| 文件 | 状态 | 功能 |
|---|---|---|
| `csrc/asu_hbm_index_maintain_aicpu/README.md` | A | maintain AICPU bring-up 说明，记录当前可运行路径是 ASU direct AICPU `.so`，vllm-ascend packaged custom op 路径仍需补齐 opdef 支持。 |
| `csrc/asu_hbm_index_maintain_aicpu/asu_hbm_index_maintain_aicpu_torch_adpt.h` | A | PyTorch adapter。校验输入均为 int32，调用 `aclnnAsuHbmIndexMaintainAicpu`，以状态输入作为输出实现原地维护。 |
| `csrc/asu_hbm_index_maintain_aicpu/op_host/CMakeLists.txt` | A | maintain op 的 host / proto 编译接入配置。 |
| `csrc/asu_hbm_index_maintain_aicpu/op_host/asu_hbm_index_maintain_aicpu_def.cpp` | A | maintain op 定义。声明 `index`、`slot_to_index`、`free_slots`、`free_head`、`last_query_slots` 输入，四个状态输出，以及 `req_num`、`seed` attr。 |
| `csrc/asu_hbm_index_maintain_aicpu/op_host/asu_hbm_index_maintain_aicpu_proto.cpp` | A | maintain op shape / dtype 推导。四个状态输出分别复用对应状态输入的 shape，dtype 为 int32。 |
| `csrc/asu_hbm_index_maintain_aicpu/op_kernel/asu_hbm_index_maintain_aicpu.cpp` | A | AICPU launcher。组装 kernel 参数并启动 `asu_hbm_index_maintain_kernel`。 |
| `csrc/asu_hbm_index_maintain_aicpu/op_kernel/asu_hbm_index_maintain_aicpu_kernel.aicpu` | A | AICPU maintain kernel。把 `last_query_slots` 标为 protected slots，从 hash 起点扫描 `slot_to_index`，回收未保护 slot 到 `free_slots`，并重置对应 `index`。 |
| `csrc/asu_hbm_index_maintain_aicpu/tmp/README.md` | A | direct AICPU maintain debug wrapper 的构建、优先级和 C ABI 说明。 |
| `csrc/asu_hbm_index_maintain_aicpu/tmp/direct_maintain.py` | A | 通过 `ctypes` 加载 ASU maintain `.so`，封装为与 `torch.ops._C_ascend.asu_hbm_index_maintain_aicpu` 同签名的 Python callable。 |

## 6. 排除的测试文件

以下测试文件在 `36e15a2fdc..HEAD` 中有变更，但按本文统计口径未纳入文件清单：

| 文件 |
|---|
| `tests/ut/attention/test_attention_v1.py` |
| `tests/ut/attention/test_offload_kv_cache_v0.py` |
| `tests/ut/attention/test_offload_kv_cache_v0_carveout.py` |
| `tests/ut/attention/test_offload_kv_cache_v0_ownership.py` |
| `tests/ut/attention/test_offload_kv_cache_v0_ref_ops.py` |
| `tests/ut/attention/test_sfa_v1.py` |
| `tests/ut/attention/test_sfa_v1_compact_static.py` |
| `tests/ut/ops/test_asu_hbm_index_csrc_wiring.py` |
