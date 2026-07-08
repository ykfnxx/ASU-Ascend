# v0.1 KV Cache Offload 真实算子接入设计

> 状态：Design
> 范围：vllm-ascend SFA eager 调试路径
> 目标：在不改变原 SFA 计算流的前提下，将 v0 的 Python 旁路校验接入真实 HBM index 算子
> 使用方法：[kv-cache-offload-v0.1-custom-op-usage.md](./kv-cache-offload-v0.1-custom-op-usage.md)

## 1. 设计结论

v0.1 不再新增概念型 `lookup_offload_slot` / `maintain_offload_slot` 算子，而是接入 ASU HBM index 真实算子：

1. `lookup` 使用 `ascend-ops/asu_hbm_index_lookup` 中的 AICore custom op。
2. `maintain` 使用 `ops/asu_hbm_index_maintain_aicpu.cpp` 和 `ops/asu_hbm_index_maintain_aicpu_kernel.aicpu` 对应的 AICPU 实现语义。
3. 当前 `ascend-ops/asu_hbm_index_maintain` 是 AICore 版本，只能作为算法等价参考，不作为 v0.1 的目标 maintain 接入路径。
4. v0.1 仍是旁路校验：真实算子只维护旁路 HBM index 和旁路 KV cache，不修改传给 `npu_sparse_flash_attention` 的原始输入。

## 2. 背景

v0 已验证纯 Python 旁路校验链路：

```text
prefill KV 写入 MicroKV
decode lightning indexer 产生 topk token_pos
Python lookup 将 token_pos 映射到旁路 cache slot
miss 时从 MicroKV 同步加载 KV
旁路 cache 读取结果与原 vLLM KV cache 寻址结果对比
SFA 继续使用原始 topk_indices / block_table / kv_cache
```

v0.1 的变化是把 Python lookup / index maintain 下沉到真实 HBM index 状态机：

```text
index          [req_num, 128K]  token_pos -> slot_id
slot_to_index  [req_num, 10K]   slot_id -> token_pos
free_slots     [req_num, 2K]    本轮 miss 可分配的 free slot 池
free_head      [req_num]        已消耗的 free slot 数
query_index    [req_num, 2K]    本轮查询 token_pos
slot_out       [req_num, 2K]    lookup 返回的 slot_id
lastQuerySlots [req_num, 2K]    maintain 保护的本轮访问 slot
```

## 3. 目标

1. 在 vLLM-Ascend SFA eager decode 路径中调用真实 `asu_hbm_index_lookup`。
2. 用 AICPU maintain 维护 `index`、`slot_to_index`、`free_slots`、`free_head` 的 eviction / free-pool 回补语义。
3. 保持 v0 的旁路校验行为：校验结果只用于断言、日志和统计，不参与 attention 输出计算。
4. 明确真实算子固定容量、padding、prefill 覆盖范围、MicroKV miss 和状态一致性处理。

## 4. 非目标

v0.1 不实现：

- 旁路 cache 接入 `npu_sparse_flash_attention`。
- decode 新 token 回写 MicroKV。
- 图捕获兼容。
- CP / DSA CP / Sparse C8 / 量化 KV cache 路径。
- 异步 miss load。
- 动态 `INDEX_SIZE`、`SLOT_COUNT`、`QUERY_COUNT`。
- 使用当前 AICore `ascend-ops/asu_hbm_index_maintain` 作为最终 maintain 接入。

## 5. 真实算子契约

### 5.1 固定常量

真实算子当前使用固定布局：

| 常量 | 值 | 含义 |
|---|---:|---|
| `INDEX_SIZE` | `128 * 1024` | 单 request 可索引的最大 token_pos 范围 |
| `SLOT_COUNT` | `10 * 1024` | 单 request / layer 的旁路 cache slot 总数 |
| `RESIDENT_SLOT_COUNT` | `8 * 1024` | 初始 resident slot 数 |
| `FREE_SLOT_COUNT` | `2 * 1024` | 单轮 miss 分配池大小 |
| `QUERY_COUNT` | `2 * 1024` | 单次 lookup 的 query 数 |
| `NOT_FOUND` | `-1` | index 或 slot_to_index 空值 |

v0.1 适配层必须显式检查：

1. `token_pos` 必须在 `[0, INDEX_SIZE)`。
2. 每个 request 单次 decode 被旁路校验的有效 query 数不能超过 `QUERY_COUNT`。
3. 旁路 KV cache 容量固定使用 `SLOT_COUNT`，不再使用任意 `VLLM_ASCEND_KV_OFFLOAD_V0_CAPACITY` 值。

### 5.2 lookup AICore 算子

Python 侧目标调用：

```python
slot_out = torch.ops._C_ascend.asu_hbm_index_lookup(
    index,
    slot_to_index,
    free_slots,
    free_head,
    query_index,
    req_num,
)
```

输入输出：

| 名称 | Shape | Dtype | 方向 | 说明 |
|---|---|---|---|---|
| `index` | `[req_num, 128K]` | int32 | in/out | `token_pos -> slot_id` |
| `slot_to_index` | `[req_num, 10K]` | int32 | in/out | `slot_id -> token_pos` |
| `free_slots` | `[req_num, 2K]` | int32 | in | free slot 池 |
| `free_head` | `[req_num]` | int32 | in/out | lookup miss 后递增 |
| `query_index` | `[req_num, 2K]` | int32 | in | 查询 token_pos |
| `slot_out` | `[req_num, 2K]` | int32 | out | 每个 query 的 slot_id |

语义：

```text
for each req row:
  for query token_pos:
    if index[token_pos] != -1:
      slot_out = index[token_pos]
    else:
      slot = free_slots[free_head]
      free_head += 1
      index[token_pos] = slot
      slot_to_index[slot] = token_pos
      slot_out = slot
```

lookup 不返回 `miss_mask`。v0.1 Python 适配层不依赖 miss mask；它根据 `query_index -> slot_out` 把本轮真实 query 对应的 KV 写入旁路 cache，padding query 不参与比较。

### 5.3 maintain AICPU 算子

v0.1 maintain 目标语义来自：

```text
ops/asu_hbm_index_maintain_aicpu.cpp
ops/asu_hbm_index_maintain_aicpu_kernel.aicpu
ops/ir/asu_hbm_index_maintain_aicpu.json
```

直接库入口：

```cpp
extern "C" void asu_hbm_index_maintain_do(
    uint32_t blockDim,
    void* stream,
    void* index,
    void* slotToIndex,
    void* freeSlots,
    void* freeHead,
    void* lastQuerySlots,
    uint32_t reqNum,
    uint32_t seed);
```

后续如果包装成 framework op，公开名称应避免和当前 AICore maintain 混淆。推荐使用显式名称：

```text
_C_ascend.asu_hbm_index_maintain_aicpu
```

或在移除 AICore maintain 注册后再使用：

```text
_C_ascend.asu_hbm_index_maintain
```

AICPU maintain 语义：

```text
for each req row:
  head = free_head[req]
  if head == 0:
    continue

  protected = bitmap(lastQuerySlots[req])
  slot = Hash32(seed ^ req) % SLOT_COUNT

  while head > 0:
    index_id = slot_to_index[slot]
    if index_id != -1 and slot not in protected:
      slot_to_index[slot] = -1
      index[index_id] = -1
      head -= 1
      free_slots[head] = slot
    slot = (slot + 1) % SLOT_COUNT

  free_head[req] = 0
```

关键约束：调用 maintain 前必须保证存在足够的非 protected resident slot。否则 prototype 算法会持续扫描，无法回补 `free_head`。

## 6. v0.1 状态所有权

### 6.1 按 `(req_id, layer_id)` 独立维护

v0.1 优先采用最保守的接入方式：每个 `(req_id, layer_id)` 拥有一套独立 HBM index 状态，并以 `req_num=1` 调用真实算子。

```text
(req_id, layer_id)
  index: int32[1, 128K]
  slot_to_index: int32[1, 10K]
  free_slots: int32[1, 2K]
  free_head: int32[1]
  query_index: int32[1, 2K]
  last_query_slots: int32[1, 2K]
  offload_k_nope_cache: tensor[10K, num_kv_heads, kv_lora_rank]
  offload_k_pe_cache: tensor[10K, num_kv_heads, qk_rope_head_dim]
```

选择 `req_num=1` 的原因：

1. 避免 vLLM batch reorder 对 op row 映射的影响。
2. 避免在 v0.1 中实现 batched state pack / scatter。
3. 保持 request 生命周期、MicroKV key 和旁路 cache 状态一一对应。

后续性能优化可以把多个活跃 request 打包成 `[req_num, ...]` 后一次调用真实算子。

### 6.2 初始化

新建 `(req_id, layer_id)` 状态时：

```text
index[:] = -1
slot_to_index[:] = -1
free_slots[0, :] = [8192, 8193, ..., 10239]
free_head[0] = 0
```

prefill 写入 MicroKV 后，v0.1 需要初始化 resident window：

```text
resident_count = min(prefill_len, RESIDENT_SLOT_COUNT, INDEX_SIZE)
for token_pos in [0, resident_count):
  slot = token_pos
  index[0, token_pos] = slot
  slot_to_index[0, slot] = token_pos
  offload_k_nope_cache[slot] = prefill k_nope[token_pos]
  offload_k_pe_cache[slot] = prefill k_pe[token_pos]
```

这个 resident window 是真实 lookup / maintain 状态机的前提：

1. lookup miss 只能从 `free_slots[0..2K)` 分配。
2. maintain 通过淘汰非 protected resident slot 回补 free pool。
3. 如果完全空表启动，maintain 在 `free_head > 0` 时没有可淘汰 slot，会违反算法前提。

v0.1 使用最简单的 resident 策略：预热请求内前 `8K` 个 prefill token。resident 策略不是本阶段优化目标。

## 7. decode 接入流程

### 7.1 SFA 插入点

插入点仍在 `AscendSFAImpl.forward()` 中，位于 lightning indexer 之后、SFA 调用之前：

```text
exec_kv() 或 _sfa_preprocess_with_mlapo() 写入 kv_cache[0] / kv_cache[1]
npu_scatter_nd_update_() 写入 kv_cache[2]
if offload_v0_enabled and 含 prefill:
    persist_prefill_kv_to_microkv(...)
    init_or_update_hbm_index_state(...)

topk_indices = indexer_select_post_process(...)

if offload_v0_enabled and decode:
    validate_topk_with_real_hbm_index_ops(...)

attn_output = npu_sparse_flash_attention(... original topk_indices ...)
```

### 7.2 query_index 构造

对每个 decode request 独立处理：

1. 从 `topk_indices` 中取属于该 request 的 flattened `token_pos`。
2. 过滤 `token_pos < 0`。
3. 过滤 `token_pos >= prefill_len`，因为 v0.1 不回写 decode 新 token。
4. 过滤 `token_pos >= INDEX_SIZE`。
5. 按 flattened 顺序去重，保留第一个出现位置。
6. 如果有效 query 数为 `0`，跳过该 request / layer 的旁路校验。
7. 如果有效 query 数大于 `QUERY_COUNT`，本 request / layer 报错或禁用旁路校验，避免部分校验造成误判。
8. 如果有效 query 数小于 `QUERY_COUNT`，用第一个有效 token_pos 填充剩余位置。

padding query 只用于满足真实算子的固定 shape，不参与旁路比较、不计入统计。

### 7.3 lookup 和旁路 cache 写入

调用：

```python
slot_out = asu_hbm_index_lookup(
    state.index,
    state.slot_to_index,
    state.free_slots,
    state.free_head,
    query_index,
    req_num=1,
)
```

对真实 query 位置执行：

```text
for each unique token_pos in valid_query:
  slot = slot_out[token_pos 对应 query offset]
  record = MicroKV.get(req_id, layer_id, token_pos)
  unpack record -> k_nope, k_pe
  offload_k_nope_cache[slot] = k_nope
  offload_k_pe_cache[slot] = k_pe
```

v0.1 允许对已 resident token 重写相同 KV。这样可以避免依赖 host-side miss shadow，也避免 AICPU eviction 后 Python 侧状态不同步。

MicroKV miss 处理：

1. 如果 `token_pos < prefill_len` 但 MicroKV miss，说明 backing store 和 HBM index 状态不一致。
2. v0.1 必须显式报错，并禁用该 request / layer 后续旁路校验。
3. 不允许静默继续，因为 lookup 已经可能为该 token 分配了 slot。

### 7.4 旁路读取比较

比较逻辑继承 v0：

```text
for each original topk entry that maps to a valid query token:
  slot = query_token_pos_to_slot[token_pos]
  bypass_k_nope = offload_k_nope_cache[slot]
  bypass_k_pe = offload_k_pe_cache[slot]
  original_k_nope / original_k_pe = 按原 vLLM block_table 和 kv_cache 寻址读取
  assert_close(bypass, original)
```

被过滤的 token 不参与比较：

- `token_pos < 0`
- `token_pos >= prefill_len`
- `token_pos >= INDEX_SIZE`
- padding query

比较失败只影响调试路径，不产生错误 attention 输出。

### 7.5 maintain 调用

比较完成后，将本轮 `slot_out` 作为 `lastQuerySlots` 调用 AICPU maintain：

```text
lastQuerySlots = slot_out
maintain_aicpu(
  index,
  slot_to_index,
  free_slots,
  free_head,
  lastQuerySlots,
  req_num=1,
  seed=step_seed,
)
```

调用前 guard：

1. 如果 `free_head == 0`，跳过 maintain。
2. 如果该状态尚未完成 resident window 初始化，跳过并报错。
3. 如果本轮有效 query 数超过 `QUERY_COUNT`，不调用 maintain，因为 lookup 本身也不应执行。

maintain 只维护 HBM index 和 free pool，不清理旁路 KV cache 内容。被淘汰 slot 的 KV payload 可以保持脏数据，因为 `index` 不再指向该 slot；下一次 lookup 分配到该 slot 后，Python 会用 MicroKV 重新写入正确 KV。

## 8. 与原计算流的关系

v0.1 必须保持以下不变量：

1. 不修改传给 SFA 的 `topk_indices`。
2. 不修改传给 SFA 的 `block_table`。
3. 不修改原始 `kv_cache[0]`、`kv_cache[1]`、`kv_cache[2]`。
4. 不把旁路 cache 作为 SFA key/value 输入。
5. 所有 lookup / maintain / MicroKV load 失败都只影响旁路校验。

## 9. 组件划分

### 9.1 ASU-Ascend 算子侧

| 路径 | v0.1 角色 |
|---|---|
| `ascend-ops/asu_hbm_index_lookup/` | 真实 lookup AICore custom op 来源 |
| `ascend-ops/torch_binding_asu_hbm_index.cpp` | lookup 的 `_C_ascend` 注册参考 |
| `ops/asu_hbm_index_maintain_aicpu.cpp` | AICPU maintain launcher 源码参考 |
| `ops/asu_hbm_index_maintain_aicpu_kernel.aicpu` | AICPU maintain kernel 逻辑 |
| `ops/ir/asu_hbm_index_maintain_aicpu.json` | AICPU framework packaging IR |
| `ascend-ops/asu_hbm_index_maintain/` | AICore maintain 参考实现，不作为 v0.1 接入目标 |

### 9.2 vLLM-Ascend 侧

| 文件 | 设计职责 |
|---|---|
| `vllm_ascend/attention/offload_kv_cache_v0.py` | 新增真实算子适配层、状态管理、query_index 构造、MicroKV load、旁路比较 |
| `vllm_ascend/attention/sfa_v1.py` | 保留最小插入点，调用 offload helper |
| `csrc/asu_hbm_index_lookup/` | 在 vLLM-Ascend 源树内承载真实 lookup AICore custom op 的 host/kernel 源码 |
| `csrc/torch_binding.cpp` | 注册 `_C_ascend.asu_hbm_index_lookup`；v0.1 不注册 AICore maintain |
| `csrc/asu_hbm_index_maintain_aicpu/` | 在 vLLM-Ascend 源树内按 custom-op 格式承载 AICPU maintain 的 `op_host/`、`op_kernel/` 和 torch adapter |
| vLLM-Ascend custom-op build | 通过既有 `csrc/**/op_host/CMakeLists.txt` 扫描接入 `asu_hbm_index_lookup` 和 `asu_hbm_index_maintain_aicpu` |

### 9.3 落地修改文件范围

v0.1 实现时，代码修改应集中在 vLLM-Ascend 的调试旁路和 ASU HBM index 算子接入边界内。ASU-Ascend 当前仓库主要承载算子源码、验证脚本、文档和面向 vLLM-Ascend 的 patch；以下 `vllm_ascend/...` 路径对应目标 vLLM-Ascend 源树或后续 patch 内容。

#### 9.3.1 vLLM-Ascend 必改文件

| 文件 | 修改内容 |
|---|---|
| `vllm_ascend/attention/offload_kv_cache_v0.py` | 承载主要实现：HBM index state、resident window 初始化、`query_index` 构造、`asu_hbm_index_lookup` 调用、MicroKV load、旁路 KV cache 写入、旁路比较、AICPU maintain 调用和错误 guard |
| `vllm_ascend/attention/sfa_v1.py` | 保持最小插入点：prefill 后写 MicroKV 并初始化状态；lightning indexer 产生 `topk_indices` 后、SFA 调用前触发真实算子旁路校验；不修改原始 SFA 输入 |
| `vllm_ascend/worker/model_runner_v1.py` | 挂接 offload manager 和 request metadata：初始化 per-layer / per-request 状态，向 attention metadata 传递 `req_id`、`num_reqs` 和 manager，并处理 request 生命周期清理 |
| `csrc/asu_hbm_index_lookup/` | 从 ASU-Ascend `ascend-ops/asu_hbm_index_lookup/` 纳入 lookup 算子实现，包含 `op_host/`、`op_kernel/` 和 torch adapter |
| `csrc/torch_binding.cpp` | include lookup torch adapter，并在 `_C_ascend` library 中注册 `asu_hbm_index_lookup` schema / PrivateUse1 impl |
| `csrc/asu_hbm_index_maintain_aicpu/` | 从 ASU-Ascend `ops/` 纳入 AICPU maintain 实现，并改成 vLLM-Ascend custom-op 目录：`op_host/CMakeLists.txt`、`*_def.cpp`、`*_proto.cpp`、`op_kernel/` 和 torch adapter |

配套测试建议覆盖：

| 文件 | 覆盖重点 |
|---|---|
| `tests/ut/attention/test_offload_kv_cache_v0.py` | `query_index` 过滤、去重、padding，resident window 初始化，MicroKV miss，query 数超过 `QUERY_COUNT` 的 guard |
| `tests/ut/attention/test_sfa_v1.py` | 确认 `sfa_v1.py` 的旁路插入点存在，且仍保留原始 SFA 调用 |
| `tests/ut/ops/test_asu_hbm_index_csrc_wiring.py` | 静态确认 lookup 和 AICPU maintain 源码位于 vLLM-Ascend `csrc` 下，lookup 完成 `_C_ascend` 注册，同时确认未注册 AICore maintain |
| `tests/e2e/nightly/single_node/ops/singlecard_ops/test_asu_decode_sfa_real_ops.py` | Ascend 环境端到端校验真实 lookup / AICPU maintain 旁路路径，确认原 SFA 输出不变 |

#### 9.3.2 ASU-Ascend 算子侧修改范围

lookup 已确认使用 `ascend-ops/asu_hbm_index_lookup`，v0.1 不需要改 lookup 语义。maintain 必须接 AICPU 实现，存在两种接入方式：

| 方式 | 需要修改的文件 | 适用阶段 |
|---|---|---|
| framework op 注册 | vLLM-Ascend `csrc/asu_hbm_index_maintain_aicpu/` 新增显式 `_C_ascend.asu_hbm_index_maintain_aicpu` 注册；`csrc/torch_binding.cpp` include torch adapter 并注册 schema / PrivateUse1 impl；`op_host/CMakeLists.txt` 进入既有 custom-op 扫描 | v0.1 目标路径，避免 Python 直接加载 `.so`，并避免误绑 AICore maintain |

落到 vLLM-Ascend 仓库时，AICPU maintain 也必须使用 custom-op 目录形态，不能放成独立 CMake 小工程。目录必须提供 `op_host/CMakeLists.txt` 供 `csrc/CMakeLists.txt` 的 `op_add_subdirectory` 自动发现，AICPU kernel 源码放在 `op_kernel/` 下。公开 torch op 名称固定为：

如果采用 framework op 注册，公开名称必须避免和当前 AICore maintain 混淆。推荐名称为：

```text
_C_ascend.asu_hbm_index_maintain_aicpu
```

#### 9.3.3 v0.1 不应修改的文件范围

| 路径 | 原因 |
|---|---|
| `ascend-ops/asu_hbm_index_maintain/` | 这是 AICore maintain 参考实现，不作为 v0.1 目标路径 |
| `csrc/sparse_flash_attention/*` | v0.1 仍是旁路校验，不接入 SFA 计算输入 |
| `npu_sparse_flash_attention_asu` 相关 patch | 属于将旁路 slot 结果接入 attention 的后续阶段，更接近 v0.2 |

## 10. 开关与 guard

沿用 v0 调试开关，并新增真实算子 guard：

```text
VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE=1
MICROKV_SOCKET=/tmp/microkv.sock
VLLM_ASCEND_KV_OFFLOAD_V0_REAL_OPS=1
```

启用后必须检查：

| 条件 | v0.1 行为 |
|---|---|
| 非 eager 路径 | 禁用或报错 |
| 非 SFA backend | 不启用 |
| CP / DSA CP 开启 | 禁用或报错 |
| Sparse C8 / indexer scale cache / 量化 KV cache | 禁用或报错 |
| MicroKV 不可连接 | 禁用或报错 |
| lookup op 未注册 | 报错，提示检查 `ascend-ops/asu_hbm_index_lookup` 接入 |
| AICPU maintain 不可用 | 报错或只运行 lookup-only 诊断模式 |
| `prefill_len > INDEX_SIZE` | 超出 `INDEX_SIZE` 的 token 不进入真实算子校验 |
| 单 request 有效 query 数超过 `QUERY_COUNT` | 禁用该 request / layer 的旁路校验 |

## 11. 错误处理

### 11.1 MicroKV miss

`token_pos < prefill_len` 的真实 query 在 MicroKV miss 时必须视为硬错误。原因是 lookup 可能已经更新 `index`，继续运行会导致 slot 指向没有正确 KV payload 的旁路 cache。

处理策略：

1. 记录 `req_id`、`layer_id`、`token_pos`、`step`。
2. 禁用该 `(req_id, layer_id)` 后续旁路校验。
3. 在严格模式下抛异常终止调试。

### 11.2 lookup free pool 耗尽

真实 lookup 当前不返回错误码。v0.1 通过前置约束降低风险：

1. 单次 query 数不超过 `2K`。
2. 状态初始化时必须建立 `8K` resident window 或确认 prefill token 全 resident。
3. 每轮 lookup 后调用 maintain 回补 free pool。

### 11.3 maintain 算法前提不满足

AICPU maintain 需要足够的非 protected resident slot。v0.1 通过 resident window 初始化保证常规 decode step 满足该前提；如果状态不是由 v0.1 初始化，禁止调用 maintain。

## 12. 测试策略

### 12.1 静态测试

1. `ascend-ops/tests/test_static_layout.py` 继续覆盖 lookup custom-op 布局。
2. 新增文档或测试时，明确 AICore maintain 不是 v0.1 目标接入。
3. 检查 Python wrapper 使用的 op 名称不会误绑到 AICore maintain。

### 12.2 算子参考测试

复用 `ops/scripts/validate_hbm_index_ops.py`：

```bash
python3 ops/scripts/validate_hbm_index_ops.py --target lookup --req-num 2 --pattern mixed
python3 ops/scripts/validate_hbm_index_ops.py --target maintain --strict-maintain
```

maintain AICPU custom op 路径：

```bash
bash csrc/build.sh -n asu_hbm_index_maintain_aicpu
python3 ops/scripts/validate_hbm_index_ops.py \
  --target maintain \
  --maintain-op _C_ascend.asu_hbm_index_maintain_aicpu \
  --strict-maintain
```

### 12.3 Python 适配层测试

需要覆盖：

1. `topk_indices -> query_index` 的过滤、去重、padding。
2. `token_pos >= prefill_len` 不进入 lookup。
3. query 数超过 `2K` 时禁用或报错。
4. resident window 初始化后的 `index`、`slot_to_index`、`free_slots`、`free_head`。
5. MicroKV miss 时禁用该 request / layer。

### 12.4 端到端调试测试

在 Ascend 环境中运行 v0 已覆盖的 SFA eager case：

1. 原 SFA 输出不变。
2. lookup 返回的 slot 能读取到 MicroKV 写入的 KV。
3. maintain 后 `free_head` 回到 `0`。
4. 多 decode step 后 `index` / `slot_to_index` 没有双向映射破坏。

## 13. 后续演进

v0.1 跑通后再考虑：

1. 将 per-request `req_num=1` 调用优化为 batched `req_num=N`。
2. 为 AICPU maintain 增加正式 framework packaging，并统一 Python 调用方式。
3. 为 lookup 增加显式 miss mask 或状态诊断输出，减少重复 MicroKV load。
4. 引入 decode 新 token 回写 MicroKV。
5. 将旁路 slot 结果接入 SFA 输入，进入 v0.2 计算路径验证。
