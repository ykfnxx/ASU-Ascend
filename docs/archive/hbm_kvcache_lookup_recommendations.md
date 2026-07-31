# HBM KVCache Lookup/Update 设计建议

本文基于当前 `simu/hbm_lookup_update` prototype 的 profiling 结果，以及 `vllm-ascend` DSA/SFA 源码分析，整理 NPU 侧 HBM KVCache 管理索引的设计建议。

目标背景：在 vLLM + vLLM-Ascend 部署形态下，为 DeepSeek V3/V3.2 类 DSA attention 构建一套 NPU HBM 上的 KVCache 管理机制。该机制需要记录 token/block 的 KVCache 是否仍在 HBM，miss 的部分由 NPU 直驱后端存储读取，并更新 HBM 中的 KVCache 与索引，从而在相同 HBM 容量下支持更高并发。

## 当前 Prototype 观察

当前 lookup kernel 的逻辑是：

- 每个 req 一个 `INDEX_SIZE = 128K` 的 `table_states`。
- 查询输入是 indexer 模拟输出的 logical token id。
- 输出是 `table_states[req, token_id]`。
- `QUERY_TILE = 64`，blockDim 将 `(req, query tile)` 分发到多个 AI Core。

关键代码：

- `../simu/hbm_lookup_update/src/hbm_lookup_update_kernel.cpp:6`
- `../simu/hbm_lookup_update/src/hbm_lookup_update_kernel.cpp:100`
- `../simu/hbm_lookup_update/src/hbm_lookup_update_kernel.cpp:112`

热点语句：

```cpp
uint32_t key = static_cast<uint32_t>(queryTile.GetValue(i));
int32_t outVal = tableStatesGm_.GetValue(indexBase + key);
outTile.SetValue(i, outVal);
```

当前实测现象：

1. `50 req * 2048 token`、`block_dim=64` 时 lookup 约 350us。
2. 去掉 `tableStatesGm_.GetValue` 后，算子降到约 34us。
3. 将索引大小从 128K 降到 8K，性能没有明显改善。

这说明主要瓶颈不是 HBM 带宽，也不是索引容量本身，而是大量 data-dependent scalar GM load 的 latency/issue 开销。

粗略量化：

```text
查询数 = 50 * 2048 = 102400
block_dim = 64
每个 core 平均查询数 ~= 102400 / 64 = 1600
350us / 1600 ~= 219ns / scalar GM load per core
有效数据量 = 102400 * 4B = 400KB
有效带宽 ~= 400KB / 350us ~= 1.17GB/s
```

1.17GB/s 远低于 HBM 带宽，因此瓶颈更像是 scalar 随机访存延迟与发射效率，而不是数据带宽。

如果按完全线性估算，`16 req * 2048` 会落在 100us 左右。这只是估算，但足以说明：独立 token 粒度 lookup 很难满足 50us 目标。

## 与 vLLM-Ascend SFA 的对比

SFA 也需要把 indexer 的 topK logical token 映射到 physical KV cache 地址：

```text
topk_token_id
  -> logical_block_id = topk_token_id / kv_cache_block_size
  -> physical_block_id = block_table[req, logical_block_id]
  -> offset = physical_block_id * block_size + offset_in_block
```

在当前 DSA 路径中：

- `sparse_block_size=1`，topK 是 token 粒度。
- KV cache block size 通常是 128。
- topK 输出按 score 排序，不按 token id/block id 排序。
- `V_TEMPLATE` 下，AIV `MergeKv` 执行 topK/block_table/KV gather，然后写入连续 `kvMergeGm_` workspace。
- AIC/Cube 等待 merge 完成后从连续 workspace 做 matmul。

因此，SFA 并没有消除 block table 查询，只是把查询与 K/V 搬运融合进 attention kernel 内部，并通过 AIV/AIC 分工和跨 loop pipeline 摊销。

这给当前项目一个很明确的判断：

```text
不要把纯 metadata lookup 做成 attention 前置独立 kernel。
```

如果独立 lookup 先花 100us 到 350us 输出 state，后续 SFA 仍然要再做一次 topK -> block_table -> physical KV 映射，那么总时延很难落入 200us 总窗口，更不用说 50us lookup 目标。

## 并发建模口径

评估 lookup 时需要区分三种并发：

1. 服务层总并发：整个推理服务正在处理的请求数。
2. DP/PP/TP 切分后的每个 rank local batch：单个 NPU rank 在某个 decode step 实际看到的 req 数。
3. 单个 kernel launch 的 `req_num`：当前仿真 lookup kernel 一次处理的 req 数。

prototype 中的 `50 req * 2048 token` 是单 kernel 的压力测试。如果真实部署通过 DP 横向扩展，系统级 50 并发不一定等于单卡一次处理 50 req。反过来，如果目标是单卡 local batch 直接达到 50，则当前 token 粒度独立 lookup 基本会成为硬瓶颈。

因此后续 benchmark 报告应始终写清：

```text
global_concurrency
dp_size / pp_size / tp_size
per_rank_req_num
query_len / sparse_count
kv_head_num
block_dim
```

尤其要注意 `topk_indices` 的 shape 包含 kv head 维度。若 `kv_head_num > 1`，SFA 内部 topK 处理和 metadata 访问会按被处理的 head 维度增加；block table 的内容虽然按 req 共享，但访问次数仍随 topK 条目数增长。

## 索引粒度建议

### 不推荐：全 token entry 保存完整 cache_entry

如果每个 token 保存完整 entry，例如：

```text
state: int32
physical_block_id: int32
offset/storage_id/epoch 等: 8B+
```

即使按 16B/token 估算：

```text
50 req * 128K token * 16B = 100MiB
```

索引本身会变得很大，而且仍然需要 token 粒度随机 scalar GM load。

### 推荐：block 粒度主索引 + token 粒度压缩状态

建议把索引拆成两层：

1. block 粒度主索引：每个 logical KV block 一个 entry。
2. token 粒度压缩状态：只保存 resident/miss/dirty/loading 等少量 bit。

以 block size 128、每 req 128K token 为例：

```text
每 req logical block 数 = 128K / 128 = 1024
50 req block entry 数 = 50 * 1024 = 51200
```

如果 block entry 是 16B：

```text
51200 * 16B = 800KiB
```

如果 token resident 状态用 bitset：

```text
50 * 128K token * 1 bit = 800KiB
50 * 128K token * 2 bit = 1.6MiB
```

这比全 token cache_entry 小很多，也更接近 vLLM 原生 block table 的访问模型。

建议的 block entry：

```cpp
struct BlockMeta {
    int32_t physical_block_id;  // HBM physical block, valid when resident
    int32_t backend_block_id;   // storage/backend block id or handle low bits
    uint32_t state_epoch;       // state bits + version/epoch
    uint32_t token_mask_ptr;    // optional offset to packed token state/mask
};
```

token state 可以按 block 压缩：

```text
2-bit/token state:
00 = not resident
01 = resident
10 = loading
11 = reserved/dirty

128 token/block * 2 bit = 256 bit = 32B/block
```

这样每个 block 的 token state 可以用 32B 表示，便于按 cacheline/向量块搬运。

## 访存与计算策略

### 1. 避免 token 粒度随机 `GetValue` 只返回 4B

当前 prototype 的核心问题是每个 topK token 都做一次：

```text
random scalar GM load -> 4B state
```

这类访问无法向量化成普通连续 DataCopy，也难以由 HBM bandwidth 吞吐来解释。即使使用 gather 指令，如果最终仍然要对每个 token 做离散 GM 地址读取，也很难从根本上改变延迟瓶颈。

### 2. 将 metadata 查询与 KV 搬运融合

参考 SFA 的 `V_TEMPLATE` 思路，更合理的模拟方向是：

```text
topK -> metadata lookup -> 判断 resident/miss
     -> resident: 直接 gather KV 到 merge workspace
     -> miss: 发起 backend read / 记录 miss list / 填充 staging slot
```

也就是说，lookup 不应只产出 `states_out`，而应该直接驱动后续的数据流：

1. resident token/block：搬运 KV 到连续 workspace；
2. miss token/block：生成后端读取请求；
3. backend load 完成：更新 HBM slot 和索引；
4. attention 使用已经整理好的连续 workspace 或已更新的 physical block。

### 3. 优先复用 block table 语义

原生 attention 依赖 `block_table` 有效。offload 后，如果某个 token 的 KV 不在 HBM，原始访问方式会失效。因此需要保证二者之一：

1. attention 前已经把 miss KV 拉回 HBM，并更新 `block_table/meta`；
2. 修改/扩展 attention 的 gather 阶段，让它在读取 KV 前检查 resident state，并在 miss 时走加载路径。

推荐方向是第二种的仿真版：扩展类似 `MergeKv` 的阶段，而不是另起纯 lookup kernel。

### 4. 不要依赖 topK 天然连续

当前 indexer 输出按 score 排名，不按 token id 或 block id 排序。局部连续性只能来自模型和数据分布，不能作为正确性或性能前提。

可以探索：

1. 对 topK 做轻量 block grouping；
2. 对同一 req/topK 内的 logical block id 做 small-cache；
3. 记录最近一次 block id 和 physical block id；
4. 对 block table entry 做 per-core UB 小缓存。

但这些都应作为实验项，而不是主设计假设。因为 topK 长度 2048，排序/grouping 本身也有成本。

## CPU 维护索引的边界

CPU 可以维护高层调度和 allocator 状态，但不适合作为每次 decode step 中 topK token resident 查询的主路径。

原因：

1. Graph 模式下 CPU 不会逐 token 参与 NPU 内部数据依赖。
2. indexer 输出在 NPU 上，SFA 消费也在 NPU 上；把 topK 拉回 CPU 查询会破坏流水。
3. 如果 NPU 直驱后端存储，load 完成与 slot 更新也发生在 NPU 可见路径上，CPU 侧状态会滞后。

建议：

```text
CPU: 维护请求级、block 分配级、换入换出策略
NPU: 维护 decode step 内必须即时可见的 resident/miss/physical slot shadow metadata
```

CPU 与 NPU 的一致性可以按 step 边界同步，而不是在每个 topK 查询中同步。

## Update 路径建议

update 不应仅仅模拟 `tableStatesGm_.SetValue(indexBase + key, newVal)`。最终需要模拟的是状态机：

```text
MISS
  -> issue_backend_read
  -> LOADING
  -> backend_done
  -> allocate_or_reuse_hbm_slot
  -> write KV payload
  -> update physical_block_id / token state / epoch
  -> RESIDENT
```

对仿真阶段，可以先做三类 update：

1. `mark_loading(topK miss list)`：把 token/block state 标记为 loading；
2. `install_resident(load_complete list)`：写 physical block id，并设置 resident bit；
3. `evict(victim list)`：清 resident bit，保留 backend id。

为了避免并发可见性问题，建议每个 metadata entry 带 epoch/version。attention/gather 使用 metadata 时可以检查 epoch 是否匹配当前 step。

## Benchmark 建议

继续保留当前 lookup benchmark，但需要把它定位为“metadata 随机 scalar load 压力测试”，而不是最终架构原型。

建议增加以下实验矩阵：

| 实验 | 目的 |
| --- | --- |
| random token state | 当前最坏形态，评估 scalar GM load 上限成本 |
| fixed address state | 判断单地址重复访问是否能被 cache/流水隐藏 |
| sequential token state | 判断连续 scalar load 与随机 scalar load 差异 |
| block state `key / 128` | 模拟 block 粒度 metadata |
| bitset state | 模拟 token state 压缩读取 |
| fused merge dummy payload | 模拟 SFA `MergeKv`，metadata 查询后搬运一行 dummy KV |
| req sweep 4/8/16/50 | 找到单卡 local batch 的线性区间 |
| block_dim sweep 32/64/128 | 判断 AI Core 并行是否饱和 |

关键 profiler 指标：

1. `aiv_scalar_time`：scalar GM lookup 是否仍是主导；
2. `aiv_mte2_time`：GM -> UB 搬运是否开始占比上升；
3. `aiv_vec_time`：是否真的有向量计算；
4. 总 `aiv_time` 随 req/query 规模是否线性；
5. 去掉 `GetValue` 的空跑时间，作为 launch/loop/output baseline。

## 推荐迭代路线

### 阶段 1：确认 metadata lookup 下界

保留当前算子，完成以下测量：

```text
req_num = 4, 8, 16, 50
query_len = 2048
block_dim = 64
pattern = random / fixed / sequential / block_id
```

目标是确认 350us 是否随查询数量线性，以及 block 粒度是否能显著降低 `aiv_scalar_time`。

### 阶段 2：实现 block metadata 仿真

新增 block 粒度 metadata：

```text
block_meta[req, logical_block_id] -> physical_block_id + block_state
token_state_bitset[req, logical_block_id, packed_word]
```

查询时：

```text
logical_block_id = token_id / 128
offset = token_id % 128
block_state = block_meta[req, logical_block_id]
token_state = bitset_test(req, logical_block_id, offset)
```

注意：如果仍然每 token 随机读 block_meta，scalar load 次数并不会自动下降。收益来自更小 working set、bitset 压缩、潜在 cache 命中，以及后续和 KV 搬运融合。

### 阶段 3：实现 fused gather 仿真

将 lookup 改造成类似 SFA `MergeKv` 的仿真：

```text
input: topK token ids, metadata, dummy KV tensor
output:
  resident workspace: 连续 dummy KV
  miss list: token/block ids
  stats: resident_count/miss_count
```

该阶段的目标不是实现真实后端 IO，而是评估“metadata 查询 + 有效数据搬运”绑定后的摊销水平。

### 阶段 4：设计真实 offload 接口

在 NPU 可见 metadata 中保存：

```text
resident bit
loading bit
physical block id / staging slot id
backend object id / offset
epoch
```

Miss 流程：

```text
topK miss
  -> compact miss list
  -> call backend read interface
  -> write HBM staging slot
  -> update metadata
  -> continue/redo gather
```

这里要和后端接口提供方确认：

1. NPU 直驱读是否支持 batch list；
2. IO completion 如何通知 kernel/graph；
3. HBM staging slot 如何分配与回收；
4. 失败、超时、重复 miss 的语义如何处理。

## 最重要的设计判断

1. token 粒度 offload 是容量目标所需，但 token 粒度完整 entry 不是好索引形式。
2. `state = table[req, token_id]` 这种纯 lookup 算子很难作为最终路径。
3. 应将 resident state 与 physical block id 合并建模，并尽量靠近 SFA 的 KV gather/merge 阶段。
4. 先让仿真从“纯 metadata 查询”升级为“metadata 查询 + KV 搬运”的 fused benchmark，否则测到的 350us 很可能只是一个无法摊销的上界。
5. 如果必须做独立查询，目标应该从 token state tensor 改成 compact miss list，减少输出量并让后续 IO 直接消费。
