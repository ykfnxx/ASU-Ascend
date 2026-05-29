# hbm_lookup_update

这是一个用于 Ascend 910B / CANN 的 HBM resident KVCache 索引仿真算子 demo。当前重点不是完整缓存系统，而是摸底两个 NPU 算子的开销：

- `lookup`：在 HBM 中维护的 per-request `key -> state` 索引里查询 `query_keys`。
- `update`：按一定比例把命中的 key 对应 state 更新为 `new_states`。

每个 req 独立拥有一张长度为 2048 的索引表。不同 req 之间的 key/state 表互不共享。

## 语义

算子不会在 kernel 内部生成 query key，调用方需要传入 `query_keys`。

多 req 输入格式：

- `table_keys`：`torch.int32` NPU tensor，形状 `[R, 2048]`，每个 req 的 resident HBM key 表。
- `table_states`：`torch.int32` NPU tensor，形状 `[R, 2048]`，每个 key 对应的 state，update 会原地修改它。
- `query_keys`：`torch.int32` NPU tensor，形状 `[R, Q]`，外部传入的查询 key。
- `new_states`：`torch.int32` NPU tensor，形状 `[R, Q]`，update 选中某个 query 位置时写入的新 state。

单 req 输入 `[2048]`、`[2048]`、`[Q]`、`[Q]` 仍然支持，会被当作 `R=1`。

Python 调用：

```python
states_out = hbm_lookup_update.lookup_random_update(
    table_keys, table_states, query_keys, new_states,
    seed=42, update_percent=5, block_dim=8, not_found=-1,
)
```

也可以分开调用，便于分别测试 lookup 和 update：

```python
states_out = hbm_lookup_update.lookup_only(
    table_keys, table_states, query_keys,
    block_dim=8, not_found=-1,
)

hbm_lookup_update.update_only(
    table_keys, table_states, query_keys, new_states,
    seed=42, update_percent=5, block_dim=8,
)
```

逻辑语义：

```cpp
// lookup
for r in 0..R-1:
    for i in 0..Q-1:
        states_out[r, i] = table_states[r, j]
            if table_keys[r, j] == query_keys[r, i]
            else not_found

// update，在 lookup 之后同 stream 执行
for r in 0..R-1:
    for pos in random_unique_positions(floor(Q * update_percent / 100), seed, r):
        key = query_keys[r, pos]
        if table_keys[r, j] == key:
            table_states[r, j] = new_states[r, pos]
```

`states_out` 返回 update 之前的 state。

当前假设：

- 每个 req 内 `table_keys[r, :]` 唯一；如果重复，lookup 返回表顺序里的第一个命中位置。
- key/state dtype 固定为 `int32`。
- 每个 req 的 table size 固定为 2048。

## 目录结构

```text
hbm_lookup_update/
├── CMakeLists.txt
├── cmake/
│   └── npu_lib.cmake
├── pybind/
│   └── pybind_hbm_lookup_update.cpp
├── run.sh
├── scripts/
│   ├── bench_lookup_update.py
│   ├── profile_lookup_update.py
│   └── test_lookup_update.py
└── src/
    └── hbm_lookup_update_kernel.cpp
```

## 编译和测试

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install pybind11
bash run.sh -v Ascend910B3
```

如果机器上的 SoC 字符串不同，先看实际型号：

```bash
npu-smi info
bash run.sh -v Ascend910B1
```

只编译，不跑测试：

```bash
bash run.sh -v Ascend910B3 -t
```

## 快速 bench

不抓 profile，只用 NPU event 快速测算子耗时：

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/bench_lookup_update.py \
  --mode lookup \
  --req-num 4 \
  --query-len 2048 \
  --block-dim 8 \
  --iters 100
```

快速扫 `req_num` 和 `block_dim`：

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/bench_lookup_update.py \
  --mode lookup \
  --req-num 4,8,16 \
  --query-len 2048 \
  --block-dim 8,16,32,64 \
  --iters 50
```

`--mode` 含义：

- `lookup`：只测 `lookup_only`。
- `update`：只测 `update_only`。
- `both`：测组合路径 `lookup_random_update`。

输出列为：

```text
mode req_num query_len block_dim update_percent iters device_ms_per_iter lookup_qps update_qps
```

## Profile

需要看算子内部 profile 时使用 `scripts/profile_lookup_update.py`。它同样可以分别跑 lookup/update，但会额外调用 `torch_npu.profiler` 落盘。

查看参数：

```bash
python3 scripts/profile_lookup_update.py --help
```

只 profile lookup：

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/profile_lookup_update.py \
  --mode lookup \
  --req-num 4 \
  --query-len 2048 \
  --block-dim 8 \
  --warmup 20 \
  --iters 50 \
  --profile-dir ./profile_lookup \
  --profiler-level level2 \
  --aic-metrics pipe
```

只 profile update：

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/profile_lookup_update.py \
  --mode update \
  --req-num 4 \
  --query-len 2048 \
  --update-percent 5 \
  --block-dim 8 \
  --warmup 20 \
  --iters 50 \
  --profile-dir ./profile_update \
  --profiler-level level2 \
  --aic-metrics pipe
```

脚本会打印：

- `host_ms_per_iter`：host 侧 wall time，包含 launch/sync 等开销。
- `device_ms_per_iter`：NPU event 统计的设备侧平均耗时，建模算子执行时间时优先看这个。
- `profile_dir`：profile 文件输出目录。

内部 profile 重点看 `profile_dir` 下的：

- `kernel_details.csv`：包含 kernel 名称、`Duration(us)`、`Block Dim`、AI Core 指标等。
- `trace_view.json`：timeline，可以用 TensorBoard、Chrome tracing、Perfetto 或 MindStudio 打开。

`--aic-metrics` 一次只能选择一组指标，常用值：

- `pipe`
- `memory`
- `ub`
- `arithmetic`
- `l2cache`
- `resource`

当前摸底阶段建议先跑 `pipe`，需要进一步判断瓶颈时再分别跑 `memory` / `ub`。

## Lookup 当前实现

每个 req 有一张独立的 `[2048]` key 表和 `[2048]` state 表。lookup 时，一个 AI Vector Core 会把当前 req 的 `table_keys/table_states` 搬到 UB，然后处理若干个 query tile。

当前 lookup 不是 scalar 逐元素扫描，而是 **全表向量匹配**：

```cpp
CompareScalar<int32_t, uint8_t>(cmpMask, tableKeysLocal, qk, CMPMODE::EQ, 2048);
Select<float, uint8_t>(hitFlag, cmpMask, oneFlag, 0.0f, ...);
ReduceMax<float>(reduceOut, hitFlag, reduceWork, 2048, true);
```

含义是：对每个 `qk`，用 vector 指令把它和当前 req 的 2048 个 `table_keys` 做一次 full-table compare，再通过 `Select + ReduceMax(calIndex=true)` 得到命中的 table index。最后只有取 index、取 state、写输出 tile 这几步保留少量 scalar 操作。

这个实现已经避免了旧版的 scalar mask 解析和 candidate 复查。profile 中应能看到：

- `aiv_vec_ratio` 明显上升。
- `aiv_scalar_ratio` 明显下降。

但它的算法关系仍然是：

```text
每个 query 和当前 req 的 2048 个 key 比较
```

所以它是 vectorized full-table compare，不是 hash/probe 索引。

## Tile 和 core 调度

当前常量：

```cpp
TABLE_SIZE = 2048
QUERY_TILE = 64
```

`QUERY_TILE` 表示一个 work item 处理一个 req 的 64 个 query。调度逻辑：

```cpp
queryTileNum = ceil(queryLen / QUERY_TILE);
totalTileNum = reqNum * queryTileNum;

for (tileId = coreId; tileId < totalTileNum; tileId += blockNum) {
    reqId = tileId / queryTileNum;
    reqTileId = tileId - reqId * queryTileNum;
}
```

一个 tile 对应：

```text
(req_id, query_tile_id)
```

例如：

```text
req_num = 4
query_len = 2048
QUERY_TILE = 64
```

则：

```text
queryTileNum = 32
totalTileNum = 4 * 32 = 128
```

如果 `block_dim=64`，最多 64 个 AI Vector Core 都能拿到任务，每个 core 平均处理约 2 个 tile。

判断任务数是否足够喂满 core：

```text
req_num * ceil(query_len / QUERY_TILE) >= block_dim
```

如果 `query_len <= QUERY_TILE`，每个 req 只有 1 个 query tile，此时 req 很少而 `block_dim` 很大时，会有很多 core 没活。

## TABLE_TILE 和 QUERY_TILE

`TABLE_TILE=64` 主要服务旧版 update 查找路径里的 `Compare<int32_t, uint8_t>(..., 64)`。当前 lookup 主路径已经使用 `CompareScalar(..., 2048)`，因此 lookup 对 `TABLE_TILE` 不敏感。

`QUERY_TILE=64` 主要用于：

- 决定每个 work item 包含多少 query。
- 决定输出 `outTile` 的大小。
- 决定 `DataCopy(statesOut, outTile, QUERY_TILE)` 的粒度。
- 决定 tile 数量和 core 调度粒度。

调小 `QUERY_TILE` 不会减少单个 query 的 compare 数量。因为每个 query 仍然执行一次：

```text
CompareScalar(table_keys[0:2048], qk)
```

调小 `QUERY_TILE` 只会增加 query tile 数，让调度粒度更细；调大 `QUERY_TILE` 会减少 tile 数和输出 DataCopy 次数，但 req 数少时可能不容易喂满 core。

当前建议：

- `TABLE_TILE` 先保持 64。
- `QUERY_TILE` 先保持 64。
- 如果要实验，可以单独测试 `QUERY_TILE=32/64/128`，但要注意这需要重新编译 kernel。

## Update 当前实现

update kernel 每个 req 由一个 core 顺序处理：

1. 把该 req 的 `table_keys/table_states` 搬到 UB。
2. 根据 `seed` 和 `update_percent` 选择一批 query 位置。
3. 对选中的 key 查找 table index。
4. 在 UB 中更新 state。
5. 把整个 2048 长度的 `table_states` 写回 HBM。

update 没有在同一个 req 内多 core 并行写 state，原因是要避免重复 key、写冲突和 last-writer 语义问题。当前阶段它更偏向简单可验证的仿真实现。

## 后续可能方向

当前 lookup 是“每 req 独立表 + 全表向量匹配”。这对 2048 长度索引很适合做 baseline，因为：

- 表可以完整放入 UB。
- 访问连续。
- vector core 利用率高。
- profile 稳定，方便估算算子开销。

如果未来要真正减少单个 query 的比较范围，可以考虑每 req 内 bucket 化索引：

```text
bucket_keys[req, bucket_num, bucket_size]
bucket_states[req, bucket_num, bucket_size]
```

查询时：

```text
bucket = Hash32(qk) % bucket_num
只和 bucket_keys[req, bucket, :] 做 CompareScalar
```

例如：

```text
bucket_num = 32
bucket_size = 64
```

单 query 的比较范围可以从 2048 降到 64，同时仍然保持连续 vector compare。但这需要重新设计 HBM 索引布局，并处理 bucket 冲突、overflow 和 update 插入策略。当前阶段可以先把 full-table vector compare 作为 baseline。
