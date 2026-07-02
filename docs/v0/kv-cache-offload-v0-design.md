# v0 KV Cache Offload 旁路校验设计

> 状态：v0 设计稿
> 范围：vllm-ascend SFA eager 调试路径
> 目标：验证 `lightning indexer -> lookup -> 旁路 cache 读取` 的正确性，不改变原 attention 计算流

## 1. 目标

v0 版本只做功能验证，目标是在不影响原 SFA 计算结果的前提下，验证以下链路是否正确：

```text
prefill KV 写入 MicroKV
decode lightning indexer 产生 topk token_pos
lookup 将 token_pos 映射为旁路 cache slot
miss 时从 MicroKV 同步加载 KV
旁路 cache 读取结果与原 vLLM KV cache 寻址结果对比
```

v0 不把旁路 cache 接入 `npu_sparse_flash_attention`。SFA 仍使用原始 `topk_indices`、原始 `block_table` 和原始 `kv_cache`。

## 2. 硬约束

1. 不修改原 vLLM KV cache layout、`slot_mapping`、`block_table` 语义。
2. 旁路 KV cache 是独立数据结构，不参与 SFA 计算。
3. 不实现最终 `indexer -> lookup -> SFA` 算子链路；v0 只在 SFA 前做旁路读取和正确性校验。
4. 不考虑性能，只支持 eager 调试；图捕获模式下禁用或报错。
5. cache miss 的加载同步完成，允许 CPU 参与 MicroKV 读写、D2H 和 H2D。
6. MicroKV 只作为 prefill 已生成 KV 的 backing store。
7. lookup 只覆盖 prefill 已生成 token；decode 新 token 不写回 MicroKV。
8. CP、Sparse C8、indexer scale、量化 KV cache 路径不纳入 v0。

## 3. 当前 SFA 插入点

vllm-ascend 的 SFA decode 路径中，关键顺序是：

1. `exec_kv()` 按原 `slot_mapping` 写入 `kv_cache[0]` 和 `kv_cache[1]`。
2. `npu_scatter_nd_update_()` 按原 `slot_mapping` 写入 indexer key `kv_cache[2]`。
3. `indexer_select_post_process()` 调用 lightning indexer，输出 `topk_indices`。
4. `_execute_sparse_flash_attention_process()` 调用 `npu_sparse_flash_attention()`。

v0 的旁路 lookup 和校验插在第 3 步之后、第 4 步之前：

```text
topk_indices = indexer_select_post_process(...)

if offload_v0_validation_enabled:
    validate_topk_against_bypass_cache(topk_indices, ...)

attn_output = npu_sparse_flash_attention(... original topk_indices ...)
```

旁路校验无论成功、失败或被跳过，都不得修改传给 SFA 的原始输入。

## 4. 数据所有权

### 4.1 MicroKV 记录

MicroKV 的存储粒度是：

```text
(req_id, layer_id, token_pos) -> MLA token record
```

`req_id` 使用 vLLM 请求 ID 字符串，而不是可随 batch reorder 变化的 `req_idx`。`layer_id` 从 attention layer name 解析得到。`token_pos` 是完整序列中的 logical token position。

MicroKV value 是调用方定义的 opaque bytes，v0 约定为一个完整 MLA token record：

```text
header
k_nope bytes
k_pe bytes
```

record 内必须包含足够的校验元信息：

| 字段 | 含义 |
|---|---|
| `magic` | record 格式标识 |
| `version` | record 格式版本 |
| `dtype` | `k_nope` 和 `k_pe` 的 dtype |
| `k_nope_shape` | 单 token `k_nope` shape |
| `k_pe_shape` | 单 token `k_pe` shape |
| `k_nope_nbytes` | `k_nope` payload 字节数 |
| `k_pe_nbytes` | `k_pe` payload 字节数 |
| `payload_checksum` | 可选校验字段，用于定位序列化错误 |

MicroKV 不解释 record 内容，不生成 slot，不参与 eviction。

### 4.2 旁路 cache

旁路 cache 按 `(req_id, layer_id)` 完全隔离维护：

```text
(req_id, layer_id)
  offload_slot_table: int32[128K]
  offload_k_nope_cache: tensor[capacity, num_kv_heads, kv_lora_rank]
  offload_k_pe_cache: tensor[capacity, num_kv_heads, qk_rope_head_dim]
```

`offload_slot_table` 是固定长度索引表：

```text
offload_slot_table[token_pos] -> offload_slot_id
```

约束：

1. 默认长度是 128K。
2. 初始值是 `-1`，表示该 token 当前不在旁路 cache。
3. `offload_slot_id` 只在当前 `(req_id, layer_id)` 内有效。
4. 不同 request、不同 layer 的 table、slot id 和 cache tensor 互不共享。
5. 同一个 MicroKV record load 到 NPU 后，`k_nope` 和 `k_pe` 写入两个旁路 tensor 的相同 `offload_slot_id`。

### 4.3 eviction 边界

v0 需要具备 eviction 语义，但 eviction 策略不在 vllm-ascend 框架侧定义。

框架侧只依赖旁路 cache manager 提供抽象接口：

```python
allocate_or_evict(req_id, layer_id, token_pos) -> tuple[int, int | None]
```

返回值：

| 返回值 | 含义 |
|---|---|
| `slot_id` | 可写入的新 token 旁路 cache slot |
| `evicted_token_pos` | 被该 slot 替换的旧 token；没有替换时为 `None` |

框架侧在收到返回值后只维护索引一致性：

```text
if evicted_token_pos is not None:
    offload_slot_table[evicted_token_pos] = -1

offload_slot_table[token_pos] = slot_id
offload_k_nope_cache[slot_id] = loaded_k_nope
offload_k_pe_cache[slot_id] = loaded_k_pe
```

被淘汰 token 的选择依据、容量大小、回收策略和热度信息不在本文档展开。

## 5. prefill 写入 MicroKV

prefill 阶段在原生 KV cache 已写入后，将已生成的 KV 按 decode miss load 所需格式写入 MicroKV。

### 5.1 插入点

prefill 写入应放在 `vllm_ascend/attention/sfa_v1.py` 的 `AscendSFAImpl.forward()` 中，位置是原生 KV cache 写入完成之后、lightning indexer 调用之前。

具体顺序是：

```text
exec_kv() 或 _sfa_preprocess_with_mlapo() 写入 kv_cache[0] / kv_cache[1]
npu_scatter_nd_update_() 写入 kv_cache[2]
if offload_v0_enabled and current batch contains prefill tokens:
    persist_prefill_kv_to_microkv(layer_name, kv_cache, slot_mapping, attn_metadata)
topk_indices = indexer_select_post_process(...)
```

选择这个位置的原因：

1. `kv_cache[0]` 和 `kv_cache[1]` 已经按原 `slot_mapping` 写入完成。
2. MLAPO 路径和 native 路径都能覆盖，因为 helper 从原 cache 读取，而不是依赖 `exec_kv()` 的返回值。
3. SFA 尚未执行，prefill 写入失败可以在调试路径中显式报错，不会产生错误 attention 输出。
4. 该逻辑只在 SFA layer 内执行，可以自然获得 `layer_name` 和该层的 `kv_cache`。

v0 不应直接使用 native 路径中 `exec_kv()` 的返回值作为 MicroKV 写入源。非 CP 路径下 `exec_kv()` 当前返回 `None`，MLAPO 路径也不会通过该返回值暴露完整 KV。因此 prefill 写入 helper 应使用 `slot_mapping` 从 `kv_cache[0]` / `kv_cache[1]` 读取已落入原 cache 的单 token KV：

```text
slot = slot_mapping[token_index]
k_nope = kv_cache[0].view(flat_slot_layout)[slot]
k_pe = kv_cache[1].view(flat_slot_layout)[slot]
```

只有满足以下条件的 token 才写入 MicroKV：

1. 当前 attention state 不是 `DecodeOnly` 或 `SpecDecoding`。
2. `token_pos < prefill_len[req_id]`。
3. `slot_mapping[token_index] >= 0`。
4. 当前 layer 是 SFA layer，且 v0 开关已启用。

为了支持上述过滤，`model_runner_v1.py` 需要把每个 token 的 CPU 侧 request 归属和位置传入 attention metadata 或 forward context：

```text
req_ids: list[str]
token_req_indices_cpu: int32[num_actual_tokens]
token_positions_cpu: int64[num_actual_tokens]
prefill_lens_cpu: int32[num_reqs]
```

其中 `prefill_lens_cpu` 使用每个请求的 prompt token 数，decode 阶段新增 token 不属于 MicroKV 覆盖范围。

写入范围：

1. 只写 prefill 已生成 token。
2. 每个 request、每个 SFA attention layer 独立写入。
3. decode 过程中新增 token 不回写 MicroKV。

写入 key：

```text
make_key(req_id, layer_id, token_pos, cache_type=KV_MLA_TOKEN)
```

写入 value：

```text
pack_mla_token_record(k_nope, k_pe)
```

prefill 写入不考虑性能。允许按 token D2H、CPU 序列化和 MicroKV `batch_put` 完成。写入失败时 v0 应报错或禁用该请求后续旁路校验，不能静默产生错误校验结果。

## 6. decode lookup 和 miss load

### 6.1 lookup 输入

每个 layer 的 lightning indexer 独立产生 `topk_indices`：

```text
topk_indices: [num_tokens, num_heads, topk]
```

v0 将 `topk_indices` 解释为请求内 `token_pos`。旁路 lookup 还需要知道每个 decode token 属于哪个 request，以及该 request 的 prefill 覆盖范围。

### 6.2 lookup 规则

对每个 `(decode_token, head, topk_item)`：

1. 如果 `token_pos < 0`，跳过旁路校验。
2. 如果 `token_pos >= prefill_len[req_id]`，跳过旁路校验。
3. 读取当前 `(req_id, layer_id)` 的 `offload_slot_table[token_pos]`。
4. 如果 slot id 非 `-1`，直接从旁路 cache 读取。
5. 如果 slot id 是 `-1`，进入同步 miss load。

### 6.3 miss load

miss load 的同步流程是：

```text
MicroKV batch_get(req_id, layer_id, token_pos)
deserialize MLA token record on CPU
allocate_or_evict(req_id, layer_id, token_pos)
H2D k_nope and k_pe
write offload_k_nope_cache[slot_id]
write offload_k_pe_cache[slot_id]
update offload_slot_table[token_pos]
```

如果 MicroKV miss，说明该 token 不在 prefill backing store 中。v0 对该 token 跳过旁路校验，原 SFA 路径继续执行。

如果 record 元信息与当前 layer 期望的 dtype 或 shape 不一致，v0 必须报错，因为继续比较会产生误导性结果。

## 7. 正确性校验

校验目标是证明同一个 `(req_id, layer_id, token_pos)`：

```text
MicroKV -> 旁路 cache -> offload_slot_id 读取出的 KV
```

与原路径：

```text
topk token_pos -> block_table -> original slot -> original kv_cache
```

读取出的 KV 一致。

### 7.1 原路径读取

原路径 slot 由当前 request 的原始 block table 计算：

```text
block_id = block_table[req_idx, token_pos // block_size]
block_offset = token_pos % block_size
original_slot = block_id * block_size + block_offset
```

读取：

```text
original_k_nope = kv_cache[0].view(flat_slot_layout)[original_slot]
original_k_pe = kv_cache[1].view(flat_slot_layout)[original_slot]
```

### 7.2 旁路读取

旁路 slot 来自当前 `(req_id, layer_id)` 的固定长度 table：

```text
offload_slot_id = offload_slot_table[token_pos]
bypass_k_nope = offload_k_nope_cache[offload_slot_id]
bypass_k_pe = offload_k_pe_cache[offload_slot_id]
```

`offload_slot_id` 不参与 SFA，不等于原 vLLM slot。

### 7.3 比较策略

v0 默认比较 `k_nope` 和 `k_pe` 两路 tensor。

建议先支持以下结果统计：

| 指标 | 含义 |
|---|---|
| `checked_items` | 完成比较的 topK 项数量 |
| `skipped_items` | 因 padding、超过 prefill 范围或 MicroKV miss 被跳过的项数量 |
| `loaded_items` | 本次 decode 中从 MicroKV miss load 的项数量 |
| `evicted_items` | 本次 decode 中触发 eviction 的项数量 |
| `mismatch_items` | 比较失败的项数量 |
| `max_abs_error` | 本次比较最大绝对误差 |

比较失败时 v0 应记录足够定位信息：

```text
req_id
layer_id
decode token index
head id
topk rank
token_pos
original_slot
offload_slot_id
max_abs_error
```

是否在第一处 mismatch 立即报错，或只记录统计后继续运行，由调试开关控制。默认建议第一处 mismatch 报错，避免继续运行掩盖错误。

## 8. 与原计算流的关系

v0 旁路校验不能改变原计算流：

1. 不修改传给 SFA 的 `topk_indices`。
2. 不修改传给 SFA 的 `block_table`。
3. 不修改原始 `kv_cache[0]`、`kv_cache[1]`、`kv_cache[2]`。
4. 不把旁路 cache 作为 SFA key/value 输入。
5. 校验失败不能产生错误 attention 输出；应通过显式异常或日志中断调试。

## 9. 开关和 guard

建议使用显式调试开关启用 v0：

```text
VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE=1
MICROKV_SOCKET=/tmp/microkv.sock
```

启用后必须检查：

| 条件 | v0 行为 |
|---|---|
| CUDA/NPU graph capture 开启 | 报错或禁用 v0 |
| CP / DSA CP 开启 | 报错或禁用 v0 |
| Sparse C8 indexer 开启 | 报错或禁用 v0 |
| indexer scale cache 存在 | 报错或禁用 v0 |
| MicroKV 不可连接 | 报错或禁用 v0 |
| 非 SFA attention backend | 不启用 v0 |

v0 是功能调试路径，默认关闭。

## 10. 组件划分

### 10.1 vllm-ascend 侧

建议新增一个旁路模块承载所有 v0 逻辑：

```text
vllm_ascend/attention/offload_kv_cache_v0.py
```

职责：

1. MicroKV client 封装。
2. MLA token record 序列化/反序列化。
3. `(req_id, layer_id)` 旁路 cache 对象管理。
4. `offload_slot_table` 查询和更新。
5. miss load 调度。
6. 原路径读取与旁路读取比较。

`sfa_v1.py` 只保留最小插入逻辑，避免把调试流程散落到 attention 主路径。

### 10.2 MicroKV 侧

MicroKV 可以保持 opaque value 模型。v0 只需要确认 cache type 命名：

```text
KV_MLA_TOKEN = 0
```

如果继续复用现有 `KV_ATTENTION_K = 0`，文档和客户端常量中需要明确该 type 在 v0 中表示完整 MLA token record，而不是单独的 K tensor。

## 11. 非目标

v0 不实现：

1. 旁路 cache 接入 SFA。
2. 新 block table layout。
3. NPU 上完整 `indexer -> lookup -> SFA` 算子链路。
4. 性能优化。
5. 图捕获兼容。
6. CP 路径兼容。
7. 量化和 Sparse C8 路径。
8. decode 新 token 回写 MicroKV。
9. 框架侧 eviction 策略。

## 12. 后续演进

v0 校验通过后，下一阶段再考虑：

1. 将 lookup 输出接入 SFA 或新 SFA 适配层。
2. 明确最终 offload cache layout 和 SFA 寻址契约。
3. 将 miss load 和 eviction 下沉到更接近生产路径的 cache manager。
4. 去除 CPU 参与的同步调试路径。
5. 支持图捕获、CP、量化和 scale。

## 13. 预计修改文件

### 13.1 vllm-ascend

| 文件 | 修改目的 |
|---|---|
| `vllm_ascend/attention/offload_kv_cache_v0.py` | 新增 v0 旁路模块，封装 MicroKV client、MLA token record 序列化/反序列化、prefill 写入、decode lookup、miss load、旁路 cache 管理和正确性比较。 |
| `vllm_ascend/attention/sfa_v1.py` | 在 `AscendSFAImpl.forward()` 中增加两个最小插入点：原生 KV 写完后触发 prefill 写 MicroKV；`indexer_select_post_process()` 返回后、SFA 前触发旁路 lookup 和校验。扩展 `AscendSFAMetadata`，携带 v0 所需的 request/token CPU metadata。 |
| `vllm_ascend/attention/utils.py` | 扩展 `AscendCommonAttentionMetadata`，增加 `req_ids`、`token_req_indices_cpu`、`token_positions_cpu`、`prefill_lens_cpu` 等字段，并在 `unpadded()` 中保持字段一致。 |
| `vllm_ascend/worker/model_runner_v1.py` | 在 `_prepare_inputs()` 中基于 `req_indices` 和 `positions_np` 构造 v0 CPU metadata；在构造 `AscendCommonAttentionMetadata` 时传入这些字段；在 runner 初始化或 KV cache 初始化阶段创建持久化的 v0 旁路 cache manager；调用 `set_ascend_forward_context()` 时把 manager 挂入 forward context。 |
| `vllm_ascend/ascend_forward_context.py` | 给 `set_ascend_forward_context()` 增加可选 `offload_kv_cache_v0` 参数，并把它加入 `_EXTRA_CTX.extra_attrs`，使 attention layer 能访问同一个跨 forward step 持久存在的 v0 manager。 |
| `vllm_ascend/envs.py` | 增加 v0 调试开关和 MicroKV socket 配置，例如 `VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE`、`MICROKV_SOCKET`。 |

### 13.2 MicroKV

| 文件 | 修改目的 |
|---|---|
| `MicroKV/python/microkv/client.py` | 增加 `KV_MLA_TOKEN` 常量，明确 cache type `0` 在 v0 中表示完整 MLA token record；如需要，增加 record helper 的薄封装入口。 |
| `MicroKV/python/microkv/__init__.py` | 导出 `KV_MLA_TOKEN`，便于 vllm-ascend 调试模块直接引用。 |
| `MicroKV/docs/design.md` | 补充 v0 中 `KV_MLA_TOKEN` 的语义，说明 value 存储完整 `k_nope + k_pe` record，而不是单独 K tensor。 |

### 13.3 测试

| 文件 | 修改目的 |
|---|---|
| `tests` 或 `vllm_ascend` 对应单元测试目录 | 增加 record pack/unpack、`token_pos -> offload_slot_id` table 更新、eviction 后 table 失效、MicroKV miss 跳过、shape/dtype mismatch 报错等单元测试。 |
| `MicroKV/tests/test_microkv_e2e.py` | 增加 `KV_MLA_TOKEN` value 往返测试，确认 MicroKV 仍只保存 opaque bytes，不解释 MLA record。 |
