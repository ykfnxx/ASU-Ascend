# hbm_lookup_update

这是一个用于 Ascend 910B / CANN 的 HBM resident KVCache 索引仿真算子 demo。当前目标是做概要性能摸测，不追求完整缓存系统语义。

当前模型：

- `lookup`：按每个 req 一张全量 token 索引表，从 HBM 中查询 `key -> state`。
- `update`：按一定比例把查询到的 token 对应 state 原地更新。
- 每个 req 的索引长度固定为 `128K`，模拟全量 token 粒度索引。
- `query_keys` 模拟 DSA indexer 输出的 token id，脚本里用有效偏移生成。

## 语义

多 req 输入格式：

- `table_keys`：`torch.int32` NPU tensor，形状 `[R, 128K]`。当前只为兼容接口保留，kernel 不读取它。
- `table_states`：`torch.int32` NPU tensor，形状 `[R, 128K]`。这是实际索引表，`table_states[req, token_id]` 表示该 token 的状态或位置。
- `query_keys`：`torch.int32` NPU tensor，形状 `[R, Q]`。当前按 indexer 输出的 token id 建模。
- `new_states`：`torch.int32` NPU tensor，形状 `[R, Q]`。update 选中某个 query 位置时写入的新 state。

单 req 输入 `[128K]`、`[128K]`、`[Q]`、`[Q]` 仍然支持，会被当作 `R=1`。

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
        key = query_keys[r, i]
        states_out[r, i] = table_states[r, key]

// update，在 lookup 之后同 stream 执行
for r in 0..R-1:
    for pos in random_unique_positions(floor(Q * update_percent / 100), seed, r):
        key = query_keys[r, pos]
        table_states[r, key] = new_states[r, pos]
```

`states_out` 返回 update 之前的 state。当前仿真假设输入侧保证 key 有效，因此 kernel 内不做 `not_found` 判断；`not_found` 参数只为兼容旧接口保留。后续如果要保存实际 KVCache 位置，可以把 `state` 编码成 `physical_block_id / slot / backend_state`，当前 demo 只用 `int32` 占位。

当前假设：

- key/state dtype 固定为 `int32`。
- 每个 req 的 index size 固定为 `128K`。
- 当前脚本生成的 `table_keys` 是 `arange(128K)`，但 kernel 不读取它。
- 输入侧保证 `query_keys` 有效，即 `0 <= key < 128K`。

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
  --req-num 50 \
  --query-len 2048 \
  --block-dim 64 \
  --iters 100
```

快速扫 `req_num` 和 `block_dim`：

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python3 scripts/bench_lookup_update.py \
  --mode lookup \
  --req-num 4,8,16,50 \
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
  --req-num 50 \
  --query-len 2048 \
  --block-dim 64 \
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
  --req-num 50 \
  --query-len 2048 \
  --update-percent 5 \
  --block-dim 64 \
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

## 当前实现

lookup 不再做全表 compare，也不做 UB gather。kernel 直接把 `query_keys` 当成有效 token id，从对应 req 的全量 `table_states` 中读取 state 并写入输出 tile：

```text
state = table_states[req, query_keys[req, i]]
```

这更贴近当前要摸测的 IO 形式：

- 每个 query 读取 1 个 `query_key`。
- 每个 query 按 key 从全量 128K `table_states` 读取 1 个 state。
- 每个 64-query tile 用一次 `DataCopy` 写回输出。

update 也不再把整张 state 表搬到 UB 再写回，而是对选中的 query 位置做随机写：

```text
table_states[req, query_key] = new_state
```

因此当前 profile 主要反映 128K token 粒度索引下的 GM 随机读写开销，不再反映 vector full-table compare 的开销。

## Tile 和 core 调度

当前常量：

```cpp
INDEX_SIZE = 128 * 1024
QUERY_TILE = 64
```

lookup 的一个 work item 对应一个 `(req_id, query_tile_id)`，每个 tile 处理 64 个 query。调度逻辑：

```cpp
queryTileNum = ceil(queryLen / QUERY_TILE);
totalTileNum = reqNum * queryTileNum;

for (tileId = coreId; tileId < totalTileNum; tileId += blockNum) {
    reqId = tileId / queryTileNum;
    reqTileId = tileId - reqId * queryTileNum;
}
```

判断 lookup 任务数是否足够喂满 core：

```text
req_num * ceil(query_len / 64) >= block_dim
```

例如 `req_num=50, query_len=2048` 时，`totalTileNum = 50 * 32 = 1600`，`block_dim=64` 可以充分分配任务。

update 的调度更简单：一个 req 由一个 core 顺序处理，`req_num=50` 时最多使用 50 个 core。

## 后续可能方向

当前 demo 的目的只是快速估计 token 粒度索引放在 HBM 后的 lookup/update IO 成本。后续如果要更贴近真实系统，可以在不改变接口形态的前提下逐步替换 `state` 编码：

- 用 state 表示 HBM hit / backend miss。
- 用 state 保存 `physical_block_id` 和 block 内 offset。
- 输出 miss token 列表，给 NPU 侧后端读取接口消费。
- update 在加载完成后写回新的 physical slot。

这些都可以先保持 `query_keys -> table_states` 的直接索引结构，避免在概要摸测阶段引入复杂 hash/bucket 维护逻辑。
