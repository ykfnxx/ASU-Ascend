# ASU HBM Token Lookup for Ascend 950 SIMT

本目录提供一个独立 PTA 原型，在一次 Ascend 950 SIMT kernel 中完成：

1. `token_id -> slot_id` 查询；
2. 唯一 miss 的 slot 分配；
3. victim 的双向映射失效；
4. HiSparse 风格的批次近似 LRU 更新。

算子不执行 host/device KV IO。调用方根据返回的 `miss_mask`，把
`query_token_ids[miss_mask]` 对应的数据加载到 `slot_ids[miss_mask]`。

## Python 接口

```python
slot_ids, miss_mask = module.asu_hbm_index_lookup_simt(
    token_to_slot,
    slot_to_token,
    lru_slots,
    query_token_ids,
    req_num,
    workspace=None,
)
```

所有输入 tensor 必须 contiguous，并位于同一 NPU：

| 参数 | shape | dtype | 含义与副作用 |
|---|---:|---|---|
| `token_to_slot` | `[req_num, 128K]` | `int32` | token 到 slot 的正向映射；`-1` 表示不驻留；原地更新 |
| `slot_to_token` | `[req_num, 10K]` | `int32` | slot 到 token 的反向映射；空 slot 为 `-1`；原地更新 |
| `lru_slots` | `[req_num, 10K]` | `int16` | 每行是全部 slot 的排列，顺序为 LRU 到 MRU；原地更新 |
| `query_token_ids` | `[req_num, 2K]` | `int32` | 本轮查询 token；只读 |
| `req_num` | scalar | Python `int` | request 行数 |
| `workspace` | 至少 `workspace_size(req_num)` 个元素 | `int32` | 可选 NPU scratch |

返回：

| 输出 | shape | dtype | 含义 |
|---|---:|---|---|
| `slot_ids` | `[req_num, 2K]` | `int32` | 每个 query 最终绑定的 slot；无效 token 返回 `-1` |
| `miss_mask` | `[req_num, 2K]` | `bool` | `True` 表示该位置负责一次实际 miss IO |

合法 token 范围为 `[0, 128K)`。范围外 token 被视为 padding：
`slot_id=-1, miss_mask=False`，不修改状态。

同一行内允许重复 token。重复的 resident token 返回相同 slot；重复 miss
只分配一次，只有 CAS 抢占成功的 canonical occurrence 返回
`miss_mask=True`，其他重复位置返回相同 slot 和 `False`。因此下游按
`miss_mask` 发起 IO 时，每个唯一 miss 只搬一次。

调用前必须满足以下状态不变量：

- `lru_slots` 是 `[0, 10K)` 全部 slot 的无重复排列；
- resident 映射双向一致：
  `token_to_slot[token] == slot` 且 `slot_to_token[slot] == token`；
- 正向表只包含 `-1` 或 `[0, 10K)`，反向表只包含 `-1` 或
  `[0, 128K)`；
- 调用入口不存在上一轮遗留的内部 `-2` claiming 状态。

## 近似 LRU 与淘汰

`lru_slots` 保存 slot，而不是 token 或物理字节地址。kernel 以当前
LRU-to-MRU 次序稳定划分：

```text
hit_slots       = 本轮 resident hit 对应的 slots
evictable_slots = 其余 slots，包括空 slot
```

第 `i` 个唯一 miss 使用 `evictable_slots[i]`。如果该 slot 原先有 token，
kernel 同时执行：

```text
token_to_slot[old_token] = -1
slot_to_token[victim_slot] = new_token
token_to_slot[new_token] = victim_slot
```

最后写回：

```text
lru_slots =
    untouched_stale_slots
    + newly_allocated_miss_slots
    + existing_hit_slots
```

这与 HiSparse 的 batch-LRU 语义一致：一轮内不按 query 的精确时间戳排序，
而是把本轮新加载项和命中项整体推到 MRU 区域，并保留各组在旧 LRU/
query 中的稳定相对顺序。

空 slot 也在 `lru_slots` 中。初始化时把空 slot 放在 LRU 前端，即可在
真正淘汰 resident token 前优先消耗空容量。

## Ascend 950 SIMT 映射

```text
grid.x = req_num
one AIV core = one request row
one asc_vf_call = 256 SIMT threads
```

每个 request 的线程共同完成：

1. 并行清空 slot hit flags；
2. 直接随机读取 `token_to_slot`，用 SIMT CAS 合并重复 miss；
3. 并行扫描 10K LRU slots；
4. 用每线程连续分片和 prefix count 做稳定 hit/evictable compaction；
5. 按 query 原顺序 compact 唯一 misses；
6. 并行失效 victims、安装新双向映射并写回 LRU；
7. 在同步后解析重复 miss 的最终 slot。

workspace 使用全局 NPU memory，避免依赖动态 shared memory：

```python
workspace = torch.empty(
    module.workspace_size(req_num),
    dtype=torch.int32,
    device="npu",
)
```

每个 request 需要 `31492` 个 `int32`，约 `123 KiB`。

## IO 边界

kernel 返回时，miss slot 已经完成元数据绑定，但 KV payload 尚未写入。
调用方必须完成：

```text
lookup/allocate/evict
  -> 按 miss_mask 将 query_token_ids 搬到 slot_ids
  -> stream/event wait
  -> attention 消费 slot_ids
```

IO 失败时本算子不提供自动 rollback。不能在 payload 完成前再次对同一
request 调用 lookup，不能让注意力读取新分配 slot，也不能让其他 kernel
并发修改同一 request 的三张状态表。

## 随机负载定义

验证、profile 和 benchmark 共享同一个 workload 生成器。通过
`--hit-count N` 精确设置每个 request 的命中数：

```text
hit_count  = N, 取值 [0, 2048]
miss_count = 2048 - N
```

初始 resident token 为 `[0, 10K)`。每个 request、每次 launch 都会：

1. 从 resident 集合中无放回随机抽取 `hit_count` 个 token；
2. 从非 resident 集合 `[10K, 128K)` 中无放回随机抽取
   `miss_count` 个 token；
3. 将两部分合并后随机 shuffle。

因此 miss 的 token ID 和它在 2K query 中的位置都是随机的，同时每一行
始终保持精确的 hit/miss 数量且没有重复 query。`--seed`、case id 和
request id 共同决定随机序列：相同参数可复现，不同 launch 使用不同 case
id。

## 编译

仅支持 Ascend 950。环境需要 CANN、CMake，以及同一 Python 环境中的
`torch`、`torch-npu` 和 `pybind11`：

```bash
cd pta-ops/asu_hbm_index_lookup_simt

# 950PR/950DT 的完整值为 <Chip Name>_<NPU Name>
bash scripts/check_soc_version.sh --device 0
SOC_VERSION="$(
  bash scripts/check_soc_version.sh --device 0 --value-only
)"

bash scripts/build_lookup_simt.sh \
  --cann-path /usr/local/Ascend/ascend-toolkit/latest \
  --soc-version "${SOC_VERSION}" \
  --python python3 \
  --build-dir build
```

检查脚本执行 `npu-smi info -t board -i 0`，将 `Chip Name` 和
`NPU Name` 拼为完整 `SOC_VERSION`；例如
`Ascend950PR_950_1234`。`--value-only` 只输出该值，可安全用于命令替换。

编译脚本通过 `--soc-version` 将完整型号传给 CMake，检查 Python 构建
依赖，完成 configure/build，并打印最终 Python extension 的路径。也可以
使用低层入口：

```bash
SOC_VERSION="${SOC_VERSION}" ./build.sh
```

## 测试

不需要 NPU 的 reference 和源码静态验证：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile python/*.py scripts/*.py
python3 -m pytest ../tests/test_lookup_simt_static_layout.py -q
```

Ascend 950 真机功能验证会同时比较两个输出以及三张原地状态表：

```bash
python3 scripts/validate_lookup_simt.py \
  --build-dir build \
  --device 0 \
  --req-num 2 \
  --hit-count 1536 \
  --seed 20260724
```

该例每个 request 精确包含 1536 个 hit 和 512 个随机 miss。边界场景可以
分别使用 `--hit-count 2048` 和 `--hit-count 0`。

## 单算子 profile

```bash
python3 scripts/profile_lookup_simt.py \
  --build-dir build \
  --device 0 \
  --req-num 1 \
  --hit-count 1536 \
  --warmup 10 \
  --profile-iterations 20 \
  --export-type db \
  --output-dir profiles/lookup-hit1536
```

`--output-dir` 必须不存在或为空。脚本在 profiler 启动前完成独立状态的
随机生成、H2D 和 warmup；采集区间中只有
`asu_hbm_index_lookup_simt` 调用和末尾同步。每次采集使用一份新状态，
保证所有 20 次调用仍然各有 1536 hit/512 miss。结果布局为：

```text
profiles/lookup-hit1536/
├── raw/                 # torch-npu 原始 profile
├── parsed/
│   └── ASCEND_PROFILER_OUTPUT/
└── manifest.json        # workload、环境、extension 和产物清单
```

`tensorboard_trace_handler` 使用同步解析
`analyse_flag=True, async_mode=False`。默认导出数据库；需要文本结果时传
`--export-type text`。

## Benchmark

```bash
python3 scripts/bench_lookup_simt.py \
  --build-dir build \
  --device 0 \
  --req-num 50 \
  --rounds 100 \
  --warmup 10 \
  --batch-rounds 10 \
  --hit-count 1536 \
  --seed 20260724 \
  --json-output benchmark-hit1536.json
```

benchmark 为每次 launch 预加载独立 NPU 状态，从而保持精确 hit/miss
数量；随机生成、H2D copy 和 warmup 均不计入时间。`--batch-rounds`
限制同时驻留的状态数，避免大 `req_num` 时一次预加载全部 round 导致
OOM；传 0 表示一次预加载全部 timed round。

结果同时报告 host wall time 和 NPU event time，包括每 launch、每
request、每 query 的耗时；有 miss 时还报告每 miss 的归一化耗时。
默认先运行一组 CPU reference 对比，可用 `--no-verify` 跳过。

## 非目标

- Ascend 910/310 或其他非 950 芯片；
- KV payload IO；
- host cache、RDMA 或 KVIO connector；
- 稀疏注意力计算；
- 跨 request 共享 slot；
- 精确逐访问时间戳 LRU。
