# DSA Sparse ASU Lookup 框架适配计划

> 文档状态：待实施
>
> 编写日期：2026-07-29
>
> 设计目标仓库：`vllm-ascend`
>
> 当前设计分支：`dsa-sparse-0.23-ops`
>
> 当前实现锚点：`57a9e6bb2559eb03e4e9cc3348141069e4995a50`
>
> 参考实现分支：`dev_lookup_maintain_integration`
>
> 本阶段只修改框架，不接入实际算子，不考虑
> `asu_hbm_index_maintain_aicpu`

## 1. 目标

当前 `dsa-sparse-0.23-ops` 框架围绕 13 Tensor 的
`dsa_sparse_lookup_update` 接口建立了固定全容量 plan、多 Query lane、
完整 LRU 和显式 workspace。

本阶段将框架收敛到 `asu_hbm_index_lookup` 的紧凑请求接口：

```python
slot_out, miss_out = lookup(
    index,
    slot_to_index,
    free_slots,
    free_head,
    req_pool_entries,
    query_index,
    lookup_mask,
    req_num,
)
```

框架只依赖这一抽象协议。实际 `torch.ops`、ACLNN host API、tiling 和
SIMT kernel 均不在本阶段接入。

完成本阶段后，后续新的 SIMT 算子应直接实现同一框架协议，并在算子内部
融合 Lookup 和 Maintain。后续算子实现不得再要求框架恢复当前 13 Tensor
ABI、多 Query lane 或显式 LRU workspace。

## 2. 冻结约束

### 2.1 执行约束

- 只支持 eager。
- 只支持 Decode consumer。
- 只支持 `AscendAttentionState.DecodeOnly`。
- 每个请求每个 model forward 只处理一个 decode token。
- 不支持 SpecDecoding、MTP 或其他一次验证多个 token 的路径。
- `max_tokens=N` 仍表示连续执行 N 个单-token Decode step。
- Main Payload I/O 继续使用 mock。
- 本阶段不调用任何实际 Lookup 算子。
- 本阶段不调用或模拟 `asu_hbm_index_maintain_aicpu`。

### 2.2 Cohort 约束

Cohort 设计保持不变：

```text
IndexCache cohort
├── 一份 lookup state
├── leader 产生 semantic Top-K
├── leader 每个 step 调用一次 Lookup
├── followers 不调用 Lookup
└── 所有层复用同一份 slot_out/miss_out
```

Main KV Payload 仍然逐层独立：

```text
cohort lookup result
├── layer 0 独立 Hot Main Cache + 独立 I/O
├── layer 1 独立 Hot Main Cache + 独立 I/O
└── ...
```

共享 lookup state 的前提是 cohort 内所有层使用相同的本地 slot 编号。某个
token 被解析到 slot `N` 后，cohort 内每一层都必须把本层对应的 Main KV
Payload 放入本层 Hot Cache 的 slot `N`。

### 2.3 ASU 接口形状约束

框架按下列固定容量准备状态和输入：

```text
index capacity       = 128K token
resident slot count  = 8K
free slot count      = 2K
lookup slot count    = 10K
query width          = 2K
free-head stride     = 16 int32
```

对应 Tensor：

```text
index             [max_num_seqs, 128K] int32
slot_to_index     [max_num_seqs, 10K]  int32
free_slots        [max_num_seqs, 2K]   int32
free_head         [max_num_seqs, 16]   int32

req_pool_entries  [B]       int32
query_index       [B, 2K]   int32
lookup_mask       [B, 2K]   int32

slot_out          [B, 2K]   int32
miss_out          [B, 2K]   int32
```

其中 `B` 是当前 model forward 的活跃请求数，不是 `max_num_seqs`。

## 3. 目标框架数据流

```text
ModelRunner
  |
  | request_ids
  | stable req_idx
  | query_positions[B]
  | seq_lens[B]
  | compact block_table[B, M]
  v
DSASparseStepMetadata
  |
  | req_pool_entries[B]
  | query_positions[B]
  | dense_tail_starts[B]
  | resident_tail_starts[B]
  v
cohort leader 产生 semantic Top-K[B, 2K]
  |
  | query_index[B, 2K]
  | lookup_mask[B, 2K]
  v
可注入 Lookup stub
  |
  | slot_out[B, 2K]
  | miss_out[B, 2K]
  v
框架合成 history slot 与 live-tail slot
  |
  | attention_indices[B, 2K]
  v
同一 cohort 结果分发到每一层
  |
  +--> layer 0 mock I/O -> layer 0 Hot Main Cache -> SFA
  +--> layer 1 mock I/O -> layer 1 Hot Main Cache -> SFA
  `--> ...
```

## 4. 保留的现有设计

以下部分继续保留：

- `RequestIndexManager` 及稳定 `request_id -> req_idx` 生命周期。
- 按 `skip_topk` 划分 IndexCache cohort。
- cohort leader/follower 关系。
- cohort 级 `lookup_complete` 语义。
- `DSASparseEagerContextRouter`。
- 每层独立 `DSASparseLayerBinding`。
- 每层独立 Hot Main Cache。
- 每层独立 I/O context、region 和 completion。
- 每层调用一次统一 I/O 接口。
- 每层使用本层 Hot Main Cache 调用现有 SFA。
- step 的 finish、abort 和 coordinator poison 语义。
- mock P/D ready、request admission 和 request retire 生命周期。

稳定 `req_idx` 直接作为 `req_pool_entries`。不增加 row-to-seat、row-to-row
或其他二次请求寻址。

## 5. 删除的设计

### 5.1 删除多 Query lane

删除以下概念和字段：

```text
max_query_tokens_per_request
query_lane_capacity
query_to_lane
active_plan_indices
active_request_indices
query_counts
```

不再使用：

```text
flat_query_index = req_idx * query_lane_capacity + lane
```

ModelRunner 必须验证每个请求当前 step 的 scheduled token 数恰好为 1。

### 5.2 删除固定全容量 Lookup Plan

从运行时框架路径删除：

```text
DSASparsePlanKey
DSASparseBatchMetadata
DSASparsePlan
```

删除其 Lookup 相关字段：

```text
token_capacity
request_capacity
query_lane_capacity
query_to_req_idx
query_to_lane
query_valid_mask
valid_topk_counts
topk_positions
resolved_hot_indices
miss_mask
workspace
```

当前 eager 路径不再把动态输入 scatter 到：

```text
[max_num_seqs * query_lane_capacity, index_topk]
```

也不再从固定输出 gather 回活跃 Query。

`block_table`、Hot block table 和 newest write descriptor 仍然需要，但应当
移动到独立的 step metadata，不能继续与 Lookup Plan 混合。

### 5.3 删除当前 LRU 框架状态

删除：

```text
token_to_hot
hot_to_token
lru_slots
dsa_sparse_lookup_workspace_stride()
per-request explicit SIMT workspace
```

本阶段框架只准备 ASU 形式的四类持久状态，不定义 Maintain 算法。

### 5.4 删除当前 13 Tensor 框架调用

框架不再调用：

```python
torch.ops._C_ascend.dsa_sparse_lookup_update(...)
```

`vllm_ascend/ops/dsa_sparse.py` 中当前依赖 `DSASparsePlan` 的 Torch adapter
退出框架主路径。

本阶段可以暂时保留旧 C++ 算子和 standalone 工具，第二阶段重新实现 SIMT
算子时再统一删除旧 ABI。

## 6. 新增的数据结构

### 6.1 Cohort 级 Lookup State

```python
@dataclass(frozen=True)
class DSASparseLookupState:
    cohort: DSASparseCohortKey
    index: torch.Tensor
    slot_to_index: torch.Tensor
    free_slots: torch.Tensor
    free_head: torch.Tensor
```

状态由 cohort 持有，不由 layer 持有。

请求行 reset 至少需要完成：

```text
index[req_idx]         = -1
slot_to_index[req_idx] = -1
free_slots[req_idx]    = [8K, ..., 10K)
free_head[req_idx]     = 0
```

首次进入 sparse decode 时，还需要定义 8K 初始 resident 映射的框架生命
周期。由于本阶段 I/O 和 Lookup 都是 stub，该动作只验证元数据和调用时序，
不用于宣称真实 Payload 正确性。

### 6.2 紧凑 Lookup Batch

```python
@dataclass(frozen=True)
class DSASparseLookupBatch:
    req_pool_entries: torch.Tensor
    query_index: torch.Tensor
    lookup_mask: torch.Tensor
```

验证条件：

- `req_pool_entries.shape == [B]`
- `query_index.shape == [B, 2048]`
- `lookup_mask.shape == [B, 2048]`
- 三者均连续。
- 三者均为 `int32`。
- `req_pool_entries` 必须唯一且落在 request pool 范围内。

### 6.3 Lookup Output

```python
@dataclass(frozen=True)
class DSASparseLookupOutput:
    slot_out: torch.Tensor
    miss_out: torch.Tensor
```

验证条件：

- 两个输出形状均与 `query_index` 相同。
- 两个输出均为 `int32`。
- 输出设备与输入设备一致。

### 6.4 Step Metadata

将 Lookup 输入与 I/O/SFA 所需信息分离：

```python
@dataclass(frozen=True)
class DSASparseStepMetadata:
    request_ids: tuple[Hashable, ...]
    req_pool_entries: torch.Tensor
    query_positions: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    dense_tail_starts: torch.Tensor
    resident_tail_starts: torch.Tensor
    write_global_slots: torch.Tensor
    write_destination_slots: torch.Tensor
    write_valid_mask: torch.Tensor
    hot_block_table: torch.Tensor
```

单-token 约束下，write descriptor 从 `[R, T]` 简化为 `[B]`。

## 7. Lookup 抽象修改

当前接口：

```python
class DSASparseLookupUpdateOperator(Protocol):
    def lookup_update(
        self,
        *,
        state: DSASparseResidencyState,
        plan: DSASparsePlan,
    ) -> None:
        ...
```

修改为：

```python
class DSASparseLookupOperator(Protocol):
    def lookup(
        self,
        *,
        state: DSASparseLookupState,
        batch: DSASparseLookupBatch,
    ) -> DSASparseLookupOutput:
        ...
```

本阶段提供显式未实现版本：

```python
class UnimplementedDSASparseLookupOperator:
    def lookup(self, **kwargs) -> DSASparseLookupOutput:
        raise NotImplementedError
```

测试通过依赖注入提供 fake：

```python
class FakeDSASparseLookupOperator:
    def lookup(self, *, state, batch):
        return DSASparseLookupOutput(
            slot_out=...,
            miss_out=...,
        )
```

产品框架代码不得在本阶段直接解析或调用任何 `torch.ops`。

## 8. Coordinator 修改

### 8.1 `DSASparseCohort`

当前：

```python
DSASparseCohort(
    key=...,
    leader_layer=...,
    state=DSASparseResidencyState(...),
    plans={plan_key: plan},
)
```

目标：

```python
DSASparseCohort(
    key=...,
    leader_layer=...,
    state=DSASparseLookupState(...),
)
```

删除 `plans`。

### 8.2 `DSASparseEagerStep`

目标结构：

```python
@dataclass
class DSASparseEagerStep:
    cohort: DSASparseCohort
    metadata: DSASparseStepMetadata
    lookup_output: DSASparseLookupOutput | None = None
    newest_written_layers: set[str] = field(default_factory=set)
    io_completed_layers: set[str] = field(default_factory=set)
    completed_layers: set[str] = field(default_factory=set)
```

`lookup_complete` 可以保留为派生属性：

```python
@property
def lookup_complete(self) -> bool:
    return self.lookup_output is not None
```

### 8.3 `begin_step`

删除：

- plan key 查找。
- 固定 plan tensor copy。
- `_copy_exact()`。
- plan 输入输出清零。
- `_prepare_batch_metadata(plan)`。

改为直接接收共享的紧凑 `DSASparseStepMetadata`。

### 8.4 `prepare_lookup`

leader 提交 Top-K 后构造：

```python
lookup_batch = DSASparseLookupBatch(
    req_pool_entries=step.metadata.req_pool_entries,
    query_index=semantic_topk_positions,
    lookup_mask=lookup_mask,
)

step.lookup_output = self.lookup_operator.lookup(
    state=step.cohort.state,
    batch=lookup_batch,
)
```

仍然满足：

- 每个 cohort 每个 step 只允许调用一次。
- 只有 leader 可以触发第一次 Lookup。
- follower 到达时必须已经存在 `lookup_output`。

### 8.5 `run_layer_attention`

每层读取同一份：

```text
step.lookup_output.slot_out
step.lookup_output.miss_out
```

随后调用本层 I/O 和本层 SFA。

## 9. Eager Runtime 修改

目标文件：

```text
vllm_ascend/worker/dsa_sparse_eager.py
```

### 9.1 `DSASparseEagerCohortDescriptor`

删除：

```text
plan_key
```

保留：

```text
cohort_key
layer_names
leader_layer
```

### 9.2 Runtime 创建

删除：

- `DSASparsePlanKey` 分配。
- `DSASparseBatchMetadata` 分配。
- 每 cohort 的 `DSASparsePlan` 分配。
- shared batch metadata identity 校验。

保留：

- cohort state 分配。
- cohort leader/follower。
- 每层独立 Hot Cache。
- 每层独立 mock I/O resource。
- coordinator freeze。

### 9.3 `begin_target_batch`

当前输入：

```text
request_ids
query_positions
query_counts
layer_metadata
```

目标输入：

```text
request_ids
query_positions
layer_metadata
```

必须满足：

```text
len(query_positions) == len(request_ids)
```

直接构造：

```python
req_pool_entries = torch.tensor(
    [
        coordinator.request_index(request_id)
        for request_id in request_ids
    ],
    dtype=torch.int32,
    device=device,
)
```

所有 target cohort 共享一个 step metadata 对象。删除当前
`stage_batch_metadata=cohort_index == 0` 机制。

## 10. ModelRunner 修改

目标文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

当前 DSA Sparse eager 接受：

```text
DecodeOnly
SpecDecoding
```

修改为只接受：

```text
DecodeOnly
```

增加显式校验：

```python
scheduled = num_scheduled_tokens[:num_reqs]
if any(int(count) != 1 for count in scheduled):
    raise RuntimeError(
        "DSA Sparse currently requires exactly one decode token "
        "per request per model forward."
    )
```

随后使用：

```text
request_ids[:B]
positions[:B]
```

两者逐行对应，不再传递 `query_counts`。

## 11. 配置修改

目标文件：

```text
vllm_ascend/dsa_sparse_config.py
```

### 11.1 删除

删除：

```text
max_query_tokens_per_request
_get_max_query_tokens_per_request()
```

当前实现已经拒绝 `num_speculative_tokens != 0`，但不应继续把恒为 1 的
字段传播到 cache config、plan 和内存预算。

### 11.2 强化校验

增加或冻结：

```text
speculative token count == 0
index_topk == 2048
max_model_len <= 128K
block_size 能够整除 8K 和 2K
```

### 11.3 删除可变 `device_buffer_size`

ASU 接口形状固定后，建议从用户配置中删除 `device_buffer_size`，改为框架
固定常量：

```text
resident slots = 8K
free slots     = 2K
lookup slots   = 10K
```

避免用户配置出与 Lookup state ABI 不兼容的形状。

## 12. Hot Cache 与 live-tail 修改

### 12.1 删除 per-query reserved newest slots

当前布局：

```text
device_buffer_size
+ max_query_tokens_per_request 个 reserved newest slots
```

删除：

```text
reserved_newest_slots
managed_hot_width = device_buffer_size + query_lane_capacity
```

目标布局：

```text
10K lookup slots
+ 一个独立 live-tail block
```

示意：

```python
lookup_slot_count = 10 * 1024
tail_start = lookup_slot_count
hot_stride = round_up(lookup_slot_count, block_size) + block_size
```

### 12.2 框架生成 `lookup_mask`

leader 得到 Top-K 后：

```python
valid_mask = semantic_topk_positions >= 0
tail_mask = (
    semantic_topk_positions
    >= dense_tail_starts[:, None]
)
lookup_mask = (
    valid_mask & ~tail_mask
).to(torch.int32)
```

### 12.3 框架合成最终 SFA 索引

Lookup 输出后：

```python
tail_slots = (
    resident_tail_starts[:, None]
    + semantic_topk_positions
    - dense_tail_starts[:, None]
)

attention_indices = torch.where(
    tail_mask,
    tail_slots,
    lookup_output.slot_out,
)
```

`attention_indices` 是 cohort step 共享结果。

## 13. I/O 接口修改

当前 I/O Lookup 相关参数：

```text
topk_positions
resolved_hot_indices
miss_mask
query_to_req_idx
```

修改为：

```text
query_index
slot_out
miss_out
req_pool_entries
```

删除 `query_to_req_idx`。紧凑批次第 `i` 行直接对应
`req_pool_entries[i]`。

Newest write descriptor、block table、Hot planes 和 completion 继续由 I/O
接口接收，但全部按当前活跃 `B` 构造，不再按 `max_num_seqs` 固定容量构造。

本阶段 mock I/O 只验证：

- Tensor shape。
- dtype。
- device。
- 每层调用一次。
- cohort 内每层收到相同的 `slot_out/miss_out`。

不验证真实 Payload。

## 14. SFA 修改

现有 SFA 接口要求三维 sparse indices。框架最终形成：

```text
attention_indices [B, 2048]
```

现有路径的：

```python
attention_indices.unsqueeze(1)
```

可继续得到：

```text
[B, 1, 2048]
```

因此 SFA kernel、ABI 和主调用路径不修改。

建议将 `DSASparseResolution.local_sparse_indices` 重命名为
`attention_indices`，避免继续使用“local”描述已经确定的 Hot Cache slot
编号。

## 15. 固定 HBM 预算修改

删除当前预算中的：

- `DSASparsePlan`。
- `resolved_hot_indices` 固定输出。
- `miss_mask` 固定输出。
- SIMT 显式 workspace。
- per-query reserved newest slots。

增加：

- 每 cohort 的 `index`。
- 每 cohort 的 `slot_to_index`。
- 每 cohort 的 `free_slots`。
- 每 cohort 的 `free_head`。
- 每层的 `10K lookup slots + live-tail block` Hot Main Cache。

本阶段的 per-forward Lookup 输入输出按活跃 `B` 构造，不计入 worker
生命周期固定最坏容量；若后续 Graph 要求地址稳定，应在 Graph 设计阶段
重新定义 capture bucket buffer，不能恢复当前全局固定 Plan。

## 16. 测试修改

### 16.1 删除的测试语义

删除或重写以下测试：

- 多 Query lane。
- Query reorder。
- 多 Query Top-K union。
- 多 Query duplicate miss canonicalization。
- `valid_topk_counts`。
- `active_plan_indices` scatter/gather。
- `DSASparsePlanKey` 多 shape。
- 固定 plan shared metadata identity。
- LRU 精确顺序。
- 显式 workspace 大小。
- DSA Sparse SpecDecoding。
- 当前 13 Tensor Python adapter。

### 16.2 新增框架测试

必须新增：

1. `req_pool_entries` 使用稳定且可以非连续的 request index。
2. Lookup 输入严格为 `[B, 2048]`。
3. Lookup 输入均为连续 `int32` NPU Tensor。
4. leader 每 cohort 每 step 只调用一次 Lookup。
5. follower 不调用 Lookup。
6. cohort 内所有层收到同一份 `slot_out/miss_out`。
7. padding Top-K 的 `lookup_mask` 为 0。
8. live-tail Top-K 的 `lookup_mask` 为 0。
9. history Top-K 的 `lookup_mask` 为 1。
10. history slot 与 live-tail slot 正确合成。
11. 每请求 scheduled token 数不是 1 时明确失败。
12. `SpecDecoding` 明确失败。
13. request release 后四类 state 的对应行被 reset。
14. Lookup stub 失败后 coordinator poison。
15. 某层 I/O 失败后 step 不能正常 finish。
16. 所有层完成后 step 正常 retire。

### 16.3 验证边界

本阶段测试只能证明：

- 框架 Tensor 合同。
- cohort leader/follower 调用次数。
- 请求寻址。
- live-tail mask 和索引合成。
- 每层 I/O/SFA fan-out。
- 生命周期和失败传播。

不能证明：

- 真实 A5 Lookup 正确性。
- 真实 Maintain 正确性。
- miss Payload 已写入。
- 多 step 模型数值正确性。
- SIMT 性能。

## 17. 文件级修改清单

### 必须修改

```text
vllm_ascend/attention/dsa_sparse.py
vllm_ascend/attention/dsa_sparse_io.py
vllm_ascend/worker/dsa_sparse_eager.py
vllm_ascend/worker/model_runner_v1.py
vllm_ascend/dsa_sparse_config.py
vllm_ascend/ops/dsa_sparse.py
```

### 小幅修改

```text
vllm_ascend/attention/sfa_v1.py
```

仅调整 resolution 字段和单-token/shape 校验，不修改 SFA kernel。

### 必须重写的主要单元测试

```text
tests/ut/attention/test_dsa_sparse.py
tests/ut/attention/test_dsa_sparse_eager.py
tests/ut/attention/test_dsa_sparse_io.py
tests/ut/worker/test_dsa_sparse_eager_runtime.py
tests/ut/worker/test_dsa_sparse_memory.py
tests/ut/test_dsa_sparse_config.py
tests/ut/worker/a2/test_model_runner_v1.py
```

### 本阶段不修改

```text
csrc/attention/dsa_sparse_lookup_update/**
tools/dsa_sparse_lookup_update/**
```

旧算子可以暂时留在源码树中，但框架不得再依赖其 Python adapter 或 13
Tensor ABI。

## 18. 实施顺序

### Task 1：冻结新数据合同

- 新增固定 ASU 容量常量。
- 新增 `DSASparseLookupState`。
- 新增 `DSASparseLookupBatch`。
- 新增 `DSASparseLookupOutput`。
- 新增 `DSASparseStepMetadata`。
- 修改 Lookup Protocol。

### Task 2：删除固定 Plan 和多 Query

- 删除 Plan/PlanKey/BatchMetadata 主路径。
- 删除 query lane。
- 删除 scatter/gather。
- 删除 LRU/workspace。
- 配置固定为单-token、Top-K 2K。

### Task 3：改造 eager runtime

- descriptor 删除 plan key。
- runtime 不再分配 plan。
- begin batch 构造紧凑 `req_pool_entries`。
- 所有 cohort 共享 step metadata。

### Task 4：改造 coordinator

- cohort 持有 ASU lookup state。
- leader 调用可注入 Lookup stub。
- step 保存 Lookup output。
- followers 复用 Lookup output。

### Task 5：改造 tail、I/O 和 SFA fan-out

- 框架生成 `lookup_mask`。
- 框架合成 live-tail slot。
- 修改 mock I/O ABI。
- 保持每层 I/O 和 SFA。

### Task 6：修改 ModelRunner 和内存预算

- 只接受 DecodeOnly。
- 每请求每 step 必须恰好一个 token。
- 更新 Hot Cache 布局。
- 更新固定 HBM breakdown。

### Task 7：重写框架测试

- 使用 Fake Lookup Operator。
- 验证 cohort 一次调用。
- 验证紧凑输入和每层 fan-out。
- 验证请求 reset、错误传播和 step retire。

## 19. 完成标准

本阶段完成必须同时满足：

1. 框架主路径不引用 `DSASparsePlan`、`query_to_lane`、`lru_slots` 或显式
   SIMT workspace。
2. 框架主路径不调用任何实际 `torch.ops` Lookup。
3. 每个 cohort 每个 step 只向注入的 Lookup Protocol 发起一次调用。
4. Lookup 输入为活跃请求紧凑 `[B, 2048]`。
5. follower 不调用 Lookup。
6. 每层独立 I/O 收到同一份 cohort Lookup output。
7. SFA 收到 `[B, 1, 2048]` 的最终 Hot Cache slot。
8. SpecDecoding 和非单-token step 在入口明确失败。
9. 新框架单元测试通过。
10. 旧 C++ 算子和 standalone 工具未被误称为新框架的实际实现。

## 20. 后续阶段边界

本计划完成后再进入 SIMT 算子阶段。后续算子需要：

- 实现本计划冻结的 Lookup 输入输出。
- 使用稳定 `req_pool_entries` 访问 cohort state。
- 在一个新 SIMT 算子内部完成 Lookup 和 Maintain。
- 返回 `slot_out/miss_out`。
- 不要求框架恢复 seed、单独 Maintain 调用、多 Query lane、LRU Tensor 或
  显式 workspace。

后续算子实现、A5 编译、NPU correctness、benchmark 和 profile 不属于本文
档当前框架适配里程碑。
