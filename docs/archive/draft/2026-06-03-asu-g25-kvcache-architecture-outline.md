# ASU-backed DSA Decode KVCache 管理总纲

## 目标

面向 Ascend NPU + ASU 存储后端，构建对标 Nvidia G2.5 生态位的 decode KVCache 管理机制：

1. 在 DSA 注意力架构下，承接 Lightning Indexer 输出的 topK original token ids。
2. 以 token 粒度管理 full KV 的 HBM resident / ASU offload 状态。
3. HBM hit 时直接使用常驻 token slot。
4. HBM miss 时由 NPU 直驱 ASU 读取 full KV，并安装到 HBM token slot。
5. 将解析后的 KV 地址交给 SFA，同时不得破坏 SFA attention 语义。

ASU 读存储接口由其他团队提供。本文只定义 KVCache 管理、地址解析和 SFA 对接边界。

## 基本事实

DSA 路径中有两类 KV 数据：

```text
kv_cache[2]:
  Lightning Indexer key cache。
  保持原 vLLM PA_BSND block layout。
  indexer 继续使用原始 block table。

kv_cache[0] + kv_cache[1]:
  SFA full KV payload。
  kv_cache[0] 为 latent key/value。
  kv_cache[1] 为 key_rope。
  二者必须作为同一个 token pair slot 管理。
```

Indexer 输出的 topK 是 original logical token id。当前 SFA 也把 `sparse_indices` 当 original token id 使用，不只是地址 id；它会参与 causal/window 边界判断。因此：

```text
sparse_indices 不能被改写成任意 remapped access id。
```

## 当前 SFA 接口限制

原 SFA PA 寻址模型是 block 粒度：

```text
token_id = sparse_indices[topk_i]
logical_block = token_id / block_size
offset = token_id % block_size
physical_block = block_table[req, logical_block]
addr = physical_block * block_size + offset
```

该接口只能表达 logical block 到 physical block 的映射，不能表达 token 粒度任意重排后的 per-token HBM slot。

因此，若目标是 token 粒度动态加载和任意 HBM slot 放置，仅靠 remap block table 不足以成为最终架构。

## 推荐路径

最终路径应保持 SFA 算法语义不变，只修改 SFA 的 KV gather / MergeKv 寻址部分：

```text
sparse_indices[topk_i] = original token id      # 用于 attention 语义
resolved_kv_slots[topk_i] = real HBM token slot # 用于 KV 地址
```

SFA 内部使用：

```text
orig_token_id = sparse_indices[topk_i]
slot = resolved_kv_slots[topk_i]

orig_token_id:
  继续用于 causal/window/seq length 语义。

slot:
  用于读取 kv_cache[0] 与 kv_cache[1] 的同一个 token pair slot。
```

这不是修改 attention 数学，只是把 KV 地址生成从 block table PA 寻址扩展为 token-level resolved slot 寻址。

## 架构分层

### 1. Indexer 原生层

保持不变：

```text
kv_cache[2]
original block table
npu_lightning_indexer
```

输出：

```text
indexer_topk_indices = original token ids
```

### 2. Token KV Resolver

新增 NPU 侧路径，承接 `indexer_topk_indices`：

```text
for token_id in topK:
    判断 managed historical 或 decode tail
    若 tail:
        使用原 vLLM block table / slot_mapping 定位 full KV
    若 managed:
        查询 token state
        HBM hit -> 返回 resident token slot
        HBM miss -> 通过 ASU 读取并安装到 free HBM token slot
    输出 resolved_kv_slot[token]
```

Resolver 只负责当前 step 必须即时可见的查询、加载、安装和统计。victim 选择、LRU、tail migration、free slot refill 由 CPU step 间完成。

### 3. SFA Gather Extension

SFA 继续接收 original `sparse_indices`，新增或等价传入 `resolved_kv_slots`。

只替换 KV gather 地址生成：

```text
old:
  sparse_indices -> block_table -> PA KV address

new:
  sparse_indices -> attention semantic checks
  resolved_kv_slots -> KV address
```

QK、softmax、PV、rope 语义和输出布局不变。

## 过渡方案

若短期必须完全不改 SFA，可做 PA-compatible staging：

```text
topK token -> resolver/load -> copy into temporary PA staging blocks
sparse_indices 保持 original token id
temporary block_table 指向 staging blocks
SFA unchanged
```

该方案可用于验证 ASU offload、cache policy 和 correctness，但不应作为最终目标：

1. staging 需要额外 KV 搬运。
2. block table 是 block 粒度，会浪费 scratch HBM。
3. 无法自然表达 token 粒度任意重排。

## 修改范围

需要新增：

```text
ASUFullKVCacheManager
managed token state / ASU record addr
HBM token pair slot pool
free slot buffer
touch / miss stats ring
Token KV Resolver NPU op
resolved_kv_slots metadata
```

需要修改：

```text
SFA KV gather / MergeKv 地址生成
attn_metadata 传递 resolved_kv_slots
```

不修改：

```text
Lightning Indexer
kv_cache[2] layout
indexer original block table
tail token 原始写入路径
ASU 读接口内部机制
SFA attention 数学主体
```

## 当前设计结论

本项目的主线不应是把 `resolved_hbm_loc` remap 成新的 `sfa_sparse_id`。正确边界是：

```text
original token id 保留给 SFA 语义。
resolved HBM slot 交给 SFA gather 寻址。
KVCache manager 负责 token 粒度 resident/miss/load/install。
SFA gather 负责按 resolved slot 读取 kv_cache[0/1] pair。
```

这条路径才能同时满足 token 粒度动态加载、降低 HBM 常驻 KV、提高 req 并发，以及不破坏 SFA 算法正确性。
