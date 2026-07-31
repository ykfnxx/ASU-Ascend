# A5 `dsa_sparse_lookup_update` SIMT 融合算子实现解析

> 分析日期：2026-07-31
>
> 源码仓库：`ykfnxx/vllm-ascend`
>
> 源码分支：`dsa-sparse-0.23-ops-simt-opt`
>
> 固定源码提交：`6eb6df91065b7d21cd43bd6c7759f5dd18884326`
>
> 目标平台：Ascend 950（项目内称 A5），CANN 9.1.0

## 1. 文档目的与结论

本文面向刚开始接触 Ascend C 和 SIMT 编程的开发者，结合固定源码提交解释
`dsa_sparse_lookup_update` 的接口、数据结构、Host Tiling、设备侧调度、并行算法、
同步语义和性能来源。

这个算子是一个 **Hot Cache 索引元数据算子**。它不搬运 MLA KV payload，而是在一次
设备调用内完成以下工作：

1. 查询 2K 个历史位置对应的 Hot Cache slot。
2. 对真正的 miss 去重，并从 2K free slot 中分配新 slot。
3. 同时更新 `index` 和 `slot_to_index` 两张互逆表。
4. 保护本轮 query 使用的 slot。
5. 淘汰等量旧 slot，把被淘汰 slot 补回 free list。
6. 输出 `slot_out` 和 `miss_out`，供后续逐层 I/O 和 SFA 使用。

当前性能优化的核心不是减少上述功能，而是改变这些功能的实现方式：

- 每个 request 由一个 256-thread SIMT block 协作处理；
- 协作 scratch 从 GM workspace 搬到 UB；
- 10K 个 `int32` 保护标记压缩成 320 个 `uint32` bitset；
- 两次平方复杂度前缀累加改成 warp/block scan；
- 每线程缓存 8 个 query、mask 和结果；
- 输出只在最后写一次；
- 0 miss 时直接跳过完整 maintain 阶段；
- 一个 VF 通过 grid-stride 处理超过物理 AIV 数量的 requests。

需要注意：本文关于逻辑和数据语义的结论来自固定源码与 CPU reference；关于 UB、
SIMT、warp、原子操作和同步的解释来自华为官方 CANN 文档；具体寄存器占用、spill、
指令吞吐和真实时延仍应以 CANN 编译报告及 A5 profile 为准。

## 2. 功能边界

### 2.1 算子负责的内容

代码中的 `token` 变量表示历史序列位置或 semantic index，不是词表 token ID。
框架传入的是 Indexer 生成的 `semantic_topk_positions`。[SRC-16]

算子维护的关系是：

```text
历史位置 / semantic index
              |
              | index[position]
              v
       Hot Cache physical slot
```

它同时维护反向关系：

```text
Hot Cache physical slot
              |
              | slot_to_index[slot]
              v
历史位置 / semantic index
```

正向表用于 lookup，反向表用于淘汰时找到旧位置并删除其正向映射。

### 2.2 算子不负责的内容

算子不执行：

- Main MLA KV 的 Host/Device I/O；
- Hot Cache payload copy；
- SFA；
- block table 管理；
- request admission；
- request pool index 的 Host 生命周期管理；
- per-layer Hot Cache plane 分配。

算子返回的 `slot_out` 和 `miss_out` 会传给逐层 `dsa_sparse_io`，之后才执行该层
SFA。[SRC-15][SRC-10]

因此，单算子 benchmark 只能证明 metadata path 的时延，不能直接代表完整
Indexer -> I/O -> SFA 链路性能。

## 3. Ascend C 与 SIMT 背景

### 3.1 一个工程化 Ascend C 算子的组成

当前算子包含以下层次：

| 层次 | 当前文件 | 作用 |
| --- | --- | --- |
| PyTorch schema | `csrc/torch_binding.cpp` | 注册 Torch operator ABI |
| PyTorch adapter | `dsa_sparse_lookup_update_torch_adpt.h` | 检查 tensor 并调用 ACLNN |
| OpDef | `op_host/dsa_sparse_lookup_update_def.cpp` | 声明输入输出、dtype、SoC |
| Shape inference | `op_host/dsa_sparse_lookup_update_infershape.cpp` | 推导输出 shape/dtype |
| Host Tiling | `op_host/dsa_sparse_lookup_update_tiling.cpp` | 校验 shape、设置 AIV 数、workspace 和 tiling data |
| ACLNN wrapper | `op_host/op_api/aclnn_dsa_sparse_lookup_update.*` | 暴露 `GetWorkspaceSize` 和执行接口 |
| Kernel shell | `op_kernel/dsa_sparse_lookup_update.cpp` | AIV kernel 入口并启动 SIMT VF |
| SIMT implementation | `op_kernel/arch35/dsa_sparse_lookup_update_simt.h` | 实际 lookup/update/maintain |

调用链为：

```text
TorchDSASparseLookupOperator.lookup
  -> torch.ops._C_ascend.dsa_sparse_lookup_update
  -> PyTorch C++ adapter
  -> aclnnDsaSparseLookupUpdate
  -> Host Tiling
  -> __global__ __aicore__ kernel shell
  -> asc_vf_call
  -> 256-thread SIMT VF
```

Torch schema 和 adapter 分别见 [SRC-1]、[SRC-2]，框架侧调用见 [SRC-9]。

### 3.2 AIV 与 AIC

Ascend AI Core 中常见两类逻辑执行资源：

- AIC：主要面向 Cube/矩阵计算；
- AIV：主要面向 Vector、标量和不规则访存。

当前算子没有矩阵乘法，主要操作是：

- 按 semantic index 随机读取；
- 整数比较；
- 原子 CAS/OR；
- 前缀 scan；
- 环形 slot 扫描。

因此 kernel 声明为：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
```

见 [SRC-6]。

Host Tiling 使用 `GetCoreNumAiv()` 获取可用 AIV 数，并通过 `SetBlockDim()` 设置实际
启动的逻辑核数。[SRC-4] 官方文档说明，对于纯 Vector 算子，`blockDim` 表示启动的
AIV 实例数，AIV 数量可通过 `GetCoreNumAiv()` 获取。[CANN-7]

### 3.3 SIMD 与 SIMT 混合编程

当前 kernel shell 使用普通 Ascend C `__global__ __aicore__` 入口，然后通过
`asc_vf_call` 启动 SIMT VF：

```cpp
asc_vf_call<DsaSparseLookupUpdateSimt>(
    dim3(DSA_SPARSE_SIMT_THREADS),
    ...);
```

设备侧 SIMT 入口声明为：

```cpp
__simt_vf__ __launch_bounds__(256) inline void
DsaSparseLookupUpdateSimt(...)
```

内部辅助函数使用：

```cpp
__simt_callee__
```

华为官方文档定义了这些函数层次：

- `__global__ __aicore__`：算子入口，协调 VF；
- `__simt_vf__`：通过 `asc_vf_call` 启动的线程级任务；
- `__simt_callee__`：只能由 SIMT VF 或其他 SIMT callee 调用的辅助函数；
- SIMT VF 内可使用 `threadIdx`、`blockIdx`、`blockDim`、`gridDim`。[CANN-1]

官方文档将 SIMT 描述为适合不规则、含分支和动态地址访问的编程模型，而 SIMD
更适合规则连续的数据并行计算。[CANN-2]

当前 lookup/update 存在大量随机地址和条件分支，因此使用 SIMT 是符合其数据访问
特征的。

### 3.4 thread、warp 和 block

当前常量为：[SRC-5]

```text
SIMT threads per block = 256
warp size              = 32
warp count             = 8
```

可以把层次理解为：

```text
一个 request
  -> 一个 SIMT thread block
       -> 256 个 threads
            -> 8 个 warps
                 -> 每个 warp 32 个 threads
```

warp 内线程可以通过 `asc_shfl_up` 直接交换寄存器值，而不必先把中间结果写入
UB/GM。官方定义中，`asc_shfl_up(var, delta)` 返回当前 lane 前 `delta` 个 lane 的
输入值。[CANN-3]

### 3.5 GM、UB 和线程本地存储

当前算子使用三类设备侧存储：

| 存储 | 源码限定符 | 当前内容 | 作用域/生命周期 |
| --- | --- | --- | --- |
| Global Memory | `__gm__` | 持久 index、query、output | 所有核可访问，跨 kernel 持久 |
| Unified Buffer | `__ubuf__` | bitset、warp totals、共享 scalar | 单 AIV/thread block 共享 |
| 线程本地变量 | 普通局部数组/标量 | 每线程 8 个 query、mask 和 result | 单 thread/SIMT VF |

官方文档说明：

- 每个 AIV 拥有独立 UB；
- 一个 thread block 内的线程共享 UB；
- SIMT thread 有独立的寄存器和栈；
- SIMT 可通过 Data Cache 访问 GM；
- UB 总空间为 256 KB；
- 还需为预留空间和 SIMT Data Cache 留出空间。[CANN-2]

因此，不能简单把 256 KB 全部当作用户 scratch。当前算子静态 UB scratch 只有
1328 bytes，对 UB 的占用非常小。

## 4. 接口与数据结构

### 4.1 PyTorch ABI

Torch schema 为：[SRC-1]

```text
dsa_sparse_lookup_update(
    Tensor(a!) index,
    Tensor(b!) slot_to_index,
    Tensor(c!) free_slots,
    Tensor(d!) free_head,
    Tensor req_pool_entries,
    Tensor query_index,
    Tensor lookup_mask,
    int req_num
) -> (Tensor, Tensor)
```

`a!`、`b!`、`c!`、`d!` 表示前四个 tensor 会被原地修改。两个返回 tensor 分别为
`slot_out` 和 `miss_out`。

Adapter 会检查：

- 所有 tensor 都为 `int32`；
- tensor shape 符合固定 ABI；
- 所有 tensor contiguous；
- 所有 tensor 位于同一设备；
- `req_num > 0`；
- `req_num <= pool_capacity`。

然后使用 `empty_like(query_index)` 分配两个输出，并通过 `EXEC_NPU_CMD` 调用
ACLNN。[SRC-2]

### 4.2 固定尺寸

设备侧常量为：[SRC-5]

| 名称 | 数值 | 含义 |
| --- | ---: | --- |
| `INDEX_CAPACITY` | 131072 | 单 request 最大 semantic index 空间 |
| `SLOT_COUNT` | 10240 | Hot Cache 索引可寻址 slot 总数 |
| `FREE_SLOT_COUNT` | 2048 | 始终保留的空闲 slot 数 |
| `QUERY_COUNT` | 2048 | 每 request 每次 lookup 的候选数 |
| `FREE_HEAD_STRIDE` | 16 | 每 request free metadata 行宽 |
| `SIMT_THREADS` | 256 | 单 request 的协作线程数 |

正常稳态为：

```text
10240 total slots
  = 8192 occupied/resident slots
  + 2048 free slots
```

每分配 `M` 个新 slot 后，必须淘汰 `M` 个旧 slot，才能重新回到 8K occupied +
2K free。

### 4.3 持久状态 tensor

每个 request pool row 有四组状态：

| Tensor | Shape | 语义 |
| --- | --- | --- |
| `index` | `[P, 131072]` | `semantic index -> slot` |
| `slot_to_index` | `[P, 10240]` | `slot -> semantic index` |
| `free_slots` | `[P, 2048]` | 当前 free slot 列表 |
| `free_head` | `[P, 16]` | 事务状态和淘汰 cursor |

框架在 `DSASparseLookupState.allocate()` 中分配它们，并将
`free_slots[row]` 初始化为 `8192...10239`。[SRC-8]

`free_head[row]` 当前只使用：

```text
free_head[row][0]：融合事务 entry head，调用入口必须为 0
free_head[row][1]：环形淘汰 cursor
free_head[row][2:16]：当前未使用，保留固定 ABI/stride
```

融合算子不允许接手一个被其他 producer 留在中间状态的 free list。如果入口
`free_head[row][0] != 0`，设备代码会为该 request 输出 `slot=-1, miss=0` 并返回，
而不是继续消费一份部分完成的列表。[SRC-7]

### 4.4 当前 batch tensor

| Tensor | Shape | 语义 |
| --- | --- | --- |
| `req_pool_entries` | `[R]` | batch row 到持久 request pool row 的映射 |
| `query_index` | `[R, 2048]` | 每个 request 的 semantic TopK |
| `lookup_mask` | `[R, 2048]` | 对应 query 是否参与 Hot Cache lookup |

这里必须区分两个索引：

```text
req_id
  当前执行 batch 中的行号，范围 0...R-1

pool_entry = req_pool_entries[req_id]
  该请求持久状态所在的 request pool row
```

因此 batch tensor 使用：

```text
query_index[req_id][entry]
```

持久状态使用：

```text
index[pool_entry][semantic_index]
slot_to_index[pool_entry][slot]
```

请求在不同 step 中改变 batch 顺序时，只需更新 `req_pool_entries`，不需要移动完整
持久状态。

### 4.5 输出语义

输出 shape 与 `query_index` 相同：

```text
slot_out[R, 2048]
miss_out[R, 2048]
```

语义为：

- `slot_out[i][j] >= 0`：该 query 最终对应的 Hot Cache slot；
- `slot_out[i][j] == -1`：masked、非法或未解析；
- `miss_out[i][j] == 1`：该位置是该 semantic index 的 canonical miss owner；
- `miss_out[i][j] == 0`：hit、duplicate follower、masked 或非法。

同一个 semantic index 在 query 中重复出现时，所有有效 occurrence 最终得到相同
slot，但只有第一个有效 occurrence 的 `miss_out` 为 1。这确保后续 I/O 不会为重复
query 搬运多次 payload。

CPU reference 对上述状态互逆关系、唯一 miss、free list、保护 slot 和 cursor 更新做了
确定性定义。[SRC-11]

## 5. Host Tiling 与 request 调度

### 5.1 Host shape 校验

Host Tiling 固定要求：[SRC-4]

```text
index            [P, 131072]
slot_to_index    [P, 10240]
free_slots       [P, 2048]
free_head        [P, 16]
req_pool_entries [R]
query_index      [R, 2048]
lookup_mask      [R, 2048]
slot_out         [R, 2048]
miss_out         [R, 2048]
```

并要求：

```text
req_num == R
req_num <= P
```

Tiling data 只传递：

```cpp
struct DsaSparseLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
};
```

### 5.2 AIV 数量

Host 设置：

```cpp
blockDim = min(req_num, aiv_count);
```

如果：

```text
req_num = 32
aiv_count = 16
```

就启动 16 个物理 AIV block。

设备侧 SIMT VF 使用：[SRC-7]

```cpp
request_stride = gridDim.x;

for (req_id = blockIdx.x;
     req_id < req_num;
     req_id += request_stride) {
    DsaSparseLookupUpdateOneRequest(...);
}
```

分工为：

```text
block 0  -> request 0, 16
block 1  -> request 1, 17
...
block 15 -> request 15, 31
```

每个 request 始终在一个 256-thread block 内完整处理，不需要不同 AIV 共同修改同一
request row。

当一个 block 连续处理多个 request 时，会复用同一份 UB scratch。源码为 0 miss
路径增加了条件 barrier，保证前一个 request 的最后一次 protected bit 原子写完成后，
下一个 request 才清空 bitset。[SRC-7]

### 5.3 workspace

优化前，算子按 request 分配用户 GM workspace。当前 Host Tiling 只保留：

```cpp
platform.GetLibApiWorkSpaceSize()
```

所需的 Ascend C 系统 workspace，不再申请用户 GM scratch。[SRC-4]

官方文档将 workspace 区分为：

- 系统 workspace：Ascend C API 需要的内部空间；
- 用户 workspace：算子自己用于额外设备侧交换/缓存的 GM 空间。[CANN-8]

kernel ABI 仍保留 `user_workspace` 参数，但当前实现没有读取它；协作数据已经放入
静态 UB。

## 6. 单 request 的线程划分

固定 workload 为：

```text
query count = 2048
slot count  = 10240
thread count = 256
```

因此每线程负责：

```cpp
query_chunk = 2048 / 256 = 8;
slot_chunk  = 10240 / 256 = 40;
```

对应静态断言在 [SRC-5]。

每个线程先读取自己的 8 个 query 和 mask：

```cpp
int32_t query_values[8];
int32_t query_masks[8];
int32_t local_slots[8];
int32_t local_misses[8];
```

这四组数组是线程本地状态。[SRC-7]

源码意图是：

- query/mask 只从 GM 读取一次；
- 各阶段复用本地值；
- `slot_out/miss_out` 不在 GM 中先初始化再覆盖；
- 最终统一写回。

编译器通常会尽量将这类小型固定数组放入寄存器，但是否实际发生 spill 必须查看
CANN 编译报告，不能只根据 C++ 源码断言。

## 7. UB scratch 布局

Kernel shell 声明：[SRC-6]

```cpp
__ubuf__ uint32_t
    shared_scratch[DSA_SPARSE_UB_SCRATCH_WORDS];
```

布局为：

```text
protected_bits : 320 × uint32 = 1280 bytes
warp_totals    :   8 × int32  =   32 bytes
scalars        :   4 × int32  =   16 bytes
-------------------------------------------
total                           1328 bytes
```

四个共享 scalar 保存：

| Offset | 名称 | 作用 |
| ---: | --- | --- |
| 0 | `kPoolEntryScalar` | `req_pool_entries[req_id]` |
| 1 | `kFreeHeadScalar` | `free_head[row][0]` |
| 2 | `kCursorScalar` | 环形淘汰 cursor |
| 3 | `kLastVictimScalar` | 本轮最后一个 victim slot |

线程 0 从 GM 读取 request row 和 `free_head`，写入 UB scalar；其他线程通过
`asc_syncthreads()` 等待后共享这些值。

## 8. 阶段一：hit 查询与唯一 miss claim

第一轮 lookup 见 [SRC-7]。

### 8.1 masked 或非法 query

满足以下任一条件时跳过：

```text
lookup_mask == 0
semantic index < 0
semantic index >= 131072
```

对应本地输出保持：

```text
slot = -1
miss = 0
```

### 8.2 resident hit

读取：

```cpp
observed = request_index[token];
```

如果：

```text
0 <= observed < 10240
```

说明该 semantic index 已驻留：

```cpp
local_slots[entry] = observed;
ProtectSlot(protected_bits, observed);
```

### 8.3 miss claim

如果 `index[token] == -1`，多个重复 query 可能同时发现 miss。算子通过在
`index[token]` 中临时写入负数 claim 选出唯一 owner：

```cpp
desired_claim = -2 - entry;
```

例如同一个 semantic index 出现在 query entry 5 和 20：

```text
entry 5  -> claim -7
entry 20 -> claim -22
```

代码使用 `asc_atomic_cas` 更新 `index[token]`。如果较晚 entry 已先写入，较早 entry
仍可用更大的负数值替换它，最终较小 entry 获胜。

`asc_atomic_cas(address, compare, val)` 的官方语义是：当地址当前值等于
`compare` 时原子写入 `val`，并返回地址原值。它支持对 GM 或 UB 的 `int32_t` 做
CAS。[CANN-4]

claim 使用 `-2-entry` 有两个价值：

1. 不需要新增一张 `owner[token]` scratch；
2. canonical owner 与 query 输入顺序一致，不依赖 warp 调度顺序。

claim 阶段结束后执行：

```cpp
asc_threadfence_block();
asc_syncthreads();
```

官方语义为：

- `asc_threadfence_block`：使调用前的内存操作对同一 thread block 内其他线程可见；
- `asc_syncthreads`：等待当前 block 的所有 thread 到达该同步点。[CANN-6]

原子操作只解决同一地址的竞争，不代表所有线程已经结束本阶段，所以阶段边界仍需要
barrier。

## 9. 阶段二：block 级 exclusive scan

每线程统计自己 8 个 query 中有多少 canonical miss：

```cpp
local_miss_count
```

然后调用：

```cpp
miss_prefix =
    BlockExclusiveScan(local_miss_count, warp_totals);
```

例如：

```text
thread local count : [2, 0, 1, 3, ...]
exclusive prefix   : [0, 2, 2, 3, ...]
```

线程 2 的第一个 canonical miss 的全局 rank 为 2。

### 9.1 warp 内 scan

每个 warp 内使用：

```cpp
asc_shfl_up(inclusive, delta)
```

其中 `delta` 依次为 1、2、4、8、16。线程直接读取同 warp 前方 lane 的寄存器值，
完成 warp inclusive scan。[SRC-7][CANN-3]

### 9.2 warp 间 scan

每个 warp 的 lane 31 将 warp total 写入：

```text
warp_totals[warp]
```

总共只有 8 个值。warp 0 再对这 8 个 total 做一次 scan，最后合成：

```text
block exclusive prefix
```

### 9.3 与优化前的差异

优化前每个线程执行：

```cpp
for (other = 0; other < tid; ++other) {
    prefix += thread_counts[other];
}
```

一次 scan 的 scratch load 次数为：

```text
0 + 1 + ... + 255 = 32640
```

miss scan 和 victim scan各执行一次，总计约 65280 次 scratch load/request。

当前实现使用 warp shuffle、8 个 UB total 和每次 scan 两个 block barrier，避免了
平方增长。

因为每个线程负责连续的 8 个 query，`thread prefix + local rank` 仍严格保持输入
顺序。

## 10. 阶段三：canonical miss 分配

得到全局 `miss_rank` 后：

```cpp
free_offset = head_start + miss_rank;
slot = request_free_slots[free_offset];
```

当前融合协议要求入口 `head_start == 0`，因此实际消费：

```text
free_slots[0:M]
```

然后更新互逆映射：

```cpp
request_slot_to_index[slot] = token;
request_index[token] = slot;
```

并写线程本地输出：

```cpp
local_slots[entry] = slot;
local_misses[entry] = 1;
```

见 [SRC-7]。

canonical semantic index 互不重复，`miss_rank` 也互不重复，因此不同线程安装的是
不同 token 和不同 slot，不需要对该分配步骤再做 CAS。

安装结束后使用 fence + barrier，保证 duplicate follower 读取时能看到最终 slot。

## 11. 阶段四：duplicate follower 解析

canonical owner 已经将：

```text
index[token] = slot
```

写回 GM。

其他仍然 `local_slots == -1` 的有效 query 会再次读取 `index[token]`。如果得到合法
slot，就设置：

```text
local_slots = slot
local_misses = 0
```

因此同一个 semantic index 的所有 occurrence 最终得到相同 slot，但只产生一次
`miss_out=1`。[SRC-7][SRC-11]

## 12. protected bitset

### 12.1 为什么要保护

本轮返回给后续 I/O/SFA 的所有 slot 都不能被紧接着的 maintain 淘汰，包括：

- resident hit；
- 本轮新分配 slot；
- duplicate follower 对应 slot。

### 12.2 bitset 布局

10240 个 slot 使用 10240 bit：

```text
10240 / 32 = 320 uint32_t
320 × 4 bytes = 1280 bytes
```

单 slot 标记方式为：

```cpp
word = slot >> 5;
bit = 1U << (slot & 31U);
```

### 12.3 为什么使用原子 OR

不同 query 可能命中相同 slot；即使 slot 不同，也可能落在同一个 32-bit word。

普通：

```cpp
protected_bits[word] |= bit;
```

包含 read-modify-write。两个线程同时执行可能互相覆盖，丢失某个保护 bit。

当前使用：

```cpp
asc_atomic_or(protected_bits + word, bit);
```

官方文档说明 `asc_atomic_or` 可对 UB 或 GM 中的 `uint32_t` 执行原子 OR。[CANN-5]

### 12.4 与旧实现的空间差异

优化前：

```text
protected_slots[10240] × int32 = 40960 bytes
```

当前：

```text
protected_bits[320] × uint32 = 1280 bytes
```

空间缩小 32 倍，同时消除了每 request 对 40 KB GM scratch 的初始化和频繁访问。

## 13. 0 miss 快路径

duplicate follower 解析完成后：

```cpp
if (total_misses == 0) {
    write outputs;
    return;
}
```

见 [SRC-7]。

0 miss 时跳过：

- 10K slot candidate scan；
- victim count；
- victim prefix scan；
- 反向索引删除；
- free list refill；
- cursor 更新。

此路径仍会执行：

- 2K query/mask load；
- `index` lookup；
- protected bitset atomic OR；
- canonical miss count scan；
- 最终 output store。

因此，`--miss-rate 0` 或未指定 miss 参数时，测到的主要是纯 hit lookup 路径，不应
与带 maintain 的非零 miss workload 混为一谈。

## 14. 融合 maintain

有 `M` 个 canonical miss 时，算子已经从 free list 消耗 `M` 个 slot。为了恢复
8K occupied + 2K free 的稳态，需要淘汰 `M` 个旧 slot。

### 14.1 环形扫描

每线程负责 40 个 scan positions：

```cpp
scan_begin = tid * 40;
scan_end = scan_begin + 40;
slot = (cursor + position) % 10240;
```

候选 victim 必须：

```text
slot 已占用
并且
slot 没有被本轮 query 保护
```

第一遍 scan 统计每线程候选数量，然后再次使用 `BlockExclusiveScan` 计算 victim
全局 rank。[SRC-7]

### 14.2 淘汰前 M 个候选

从 cursor 开始的环形顺序中，前 `M` 个候选被选中。

淘汰时：

```cpp
old_token = request_slot_to_index[slot];
request_slot_to_index[slot] = -1;

if (request_index[old_token] == slot) {
    request_index[old_token] = -1;
}
```

随后将 victim slot 写回 free list：

```cpp
request_free_slots[M - 1 - victim_rank] = slot;
```

这里按 victim rank 逆序回填 consumed prefix，是当前 CPU reference 明确定义的
free-list 顺序。[SRC-11]

### 14.3 cursor 更新

处理结束后线程 0 设置：

```cpp
free_head[row][1] = (last_victim + 1) % 10240;
free_head[row][0] = 0;
```

下一次 maintain 从本轮最后 victim 的后一个 slot 开始。

### 14.4 为什么不是严格 LRU

当前实现没有：

- per-slot timestamp；
- hit 时更新的链表；
- 精确访问次序记录。

它实际执行：

```text
从 cursor 环形扫描
  -> 跳过本轮访问的 protected slots
  -> 选择前 M 个 occupied slots
  -> cursor 移到最后 victim 后面
```

更准确的名称是“带本轮访问保护的环形近似淘汰”，而不是严格 LRU。

## 15. 最终输出与状态不变量

每线程最终只把自己的 8 个结果写入 GM 一次：

```cpp
slot_out[offset] = local_slots[local_entry];
miss_out[offset] = local_misses[local_entry];
```

完成后应保持：

1. `index[token] = slot` 时，`slot_to_index[slot] = token`。
2. `slot_to_index[slot] = token` 时，`index[token] = slot`。
3. `free_slots` 中的 slot 在 `slot_to_index` 中必须为空。
4. `free_slots` 内没有重复 slot。
5. 入口和出口 `free_head[row][0] == 0`。
6. cursor 始终位于 `[0, 10240)`。
7. 每个 canonical miss 只有一个 `miss_out=1`。
8. 本轮输出的所有 slot 均不会在同一调用的 maintain 中被淘汰。

CPU reference 在调用前后检查这些关系。[SRC-11]

当前设备 kernel 依赖框架保证：

- 同一次调用中的 `req_pool_entries` 不重复；
- 不同 SIMT blocks 不会并发修改同一个持久 request row；
- 初始状态满足 8K occupied + 2K free；
- 正反向表一致；
- free list 合法；
- canonical miss 数不超过 2K。

CPU reference 会显式拒绝重复 pool row；device kernel 没有额外扫描
`req_pool_entries` 去检查重复，因此这是调用方约束。

## 16. 当前优化为何有效

### 16.1 scratch 从 GM 移到 UB

优化前每 request 的 GM scratch：

```text
protected_slots[10240] = 40960 bytes
thread_counts[256]     =  1024 bytes
scalars[4]             =    16 bytes
---------------------------------------
total                  = 42000 bytes
```

当前每活跃 AIV 的 UB scratch：

```text
protected_bits[320] = 1280 bytes
warp_totals[8]      =   32 bytes
scalars[4]          =   16 bytes
--------------------------------
total               = 1328 bytes
```

这同时减少：

- GM workspace 申请量；
- 每次调用的 scratch 清零量；
- GM/L2 流量；
- 协作状态访问延迟。

### 16.2 前缀计算从 O(256²) 变成分层 scan

旧实现的每线程前向累加会产生约 32640 次 scratch load/scan。当前通过 warp shuffle
和 8 个 warp total 完成 block scan，计算量随 warp 数和 `log2(warp_size)` 增长。

### 16.3 线程本地缓存

每线程一次读取 8 个 query/mask，并在本地保存 slot/miss：

- 避免三个阶段反复读 query/mask；
- 避免在 GM 中把输出先清零再覆盖；
- 最终输出只写一次。

是否发生 register spill 仍需以编译报告为准。

### 16.4 all-hit 快路径

0 miss 时跳过整个 10K maintain scan。对 hit rate 很高的 decode workload，这会显著
降低单调用时延。

### 16.5 融合 lookup 与 maintain

单 kernel 内完成：

```text
lookup
  -> unique miss allocation
  -> reciprocal map update
  -> protected-set construction
  -> maintain
```

相比两个独立算子，减少：

- 第二次 kernel launch；
- 中间状态发布；
- 独立 maintain 的调度和同步开销；
- `slot_out` 再次读取和重解析。

### 16.6 request 级 AIV 并行

`min(req_num, aiv_count)` 个 AIV 并行处理不同 request。超过物理 AIV 数的 request
通过 VF 内 grid-stride 继续处理，并复用当前 AIV 的 UB scratch。

## 17. 同步与 barrier 的准确理解

当前实现不是“完全没有 barrier”，而是只在跨线程阶段边界同步。

主要同步点包括：

1. thread 0 广播 pool entry/cursor 后；
2. miss claim 完成后；
3. warp totals 写入和 warp 0 scan 后；
4. canonical slot 安装完成后；
5. duplicate protected bits 完成后；
6. victim 删除和 last-victim 发布后；
7. 0 miss 且 scratch 将被下一 request 复用时。

`BlockExclusiveScan` 每次含两个 `asc_syncthreads()`：

- 等待 8 个 warp total 就绪；
- 等待 warp 0 的 warp-total scan 完成。

这些 barrier 是 block-wide scan 正确性的组成部分。优化重点是移除不必要的 fence、
GM scratch 和平方级 prefix，不是盲目删除所有同步。

官方文档明确区分：

- barrier：等待线程；
- fence：建立内存操作可见性顺序；
- atomic：保证单个地址的原子 read-modify-write。[CANN-4][CANN-5][CANN-6]

## 18. 框架调用位置

### 18.1 TopK 到 lookup

Leader layer 第一次得到 `semantic_topk_positions` 时调用：

```python
self.coordinator.prepare_lookup(
    self.step,
    query_index=semantic_topk_positions,
)
```

见 [SRC-16]。

`prepare_lookup`：

1. 检查 `query_index` shape/dtype/contiguous；
2. 生成 valid mask；
3. 排除仍位于 dense live tail 的位置；
4. 生成 `lookup_mask`；
5. 调用一次 `lookup_operator.lookup()`；
6. 合成 live-tail slot 与 `slot_out`，得到最终 `attention_indices`。

live tail 不进入 Hot Cache lookup，而是由框架直接计算对应 resident-tail slot。
[SRC-15]

### 18.2 cohort 共享 lookup

当前路径是：

```text
cohort leader 计算 TopK
  -> 每 cohort 每 step 做一次 lookup
  -> follower layers 复用同一份 slot/miss mapping
```

不是每个物理 layer 重复执行相同 lookup。

### 18.3 per-layer I/O

索引 mapping 共享，但每个物理 layer 仍拥有独立：

- Hot Main Cache planes；
- I/O context；
- I/O region；
- completion；
- payload。

每层调用 `dsa_sparse_io` 时传入共享的：

```text
query_index
slot_out
miss_out
req_pool_entries
```

随后使用该层 Hot Cache 执行 SFA。[SRC-15][SRC-10]

准确的所有权是：

```text
cohort：
  lookup/index/residency metadata

physical layer：
  Hot Cache payload planes
  I/O resources
  SFA invocation
```

## 19. profile 结果如何解释

### 19.1 默认 workload

profile 脚本默认：

```text
resident entries/request = 8192
query width/request       = 2048
miss count/request        = 0
```

未指定 `--miss-rate` 或 `--miss-count` 时，测的是 all-hit 快路径。[SRC-13][SRC-14]

### 19.2 miss 参数

可以使用：

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0 \
  --requests 32 \
  --miss-rate 10
```

或：

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0 \
  --requests 32 \
  --miss-count 205
```

两个参数互斥。固定 query width 为 2048，因此 10% 会按脚本规则四舍五入成
205 misses/request，实际比例约为 10.0098%。

### 19.3 Event timing

非零 miss 会修改持久 state。为了保证每次 invocation 都具有相同 miss 数量，脚本会
在每轮前恢复输入状态。

当前 Event 时序为：

```text
restore state
record start event
invoke operator
record end event
```

因此 NPU Event latency 不包含 state restore，但包含 start/end event 之间的设备流
区间。[SRC-13]

它不是：

- Python API 端到端时延；
- 完整请求时延；
- KV I/O 时延；
- SFA 时延；
- 模型层时延。

### 19.4 profiler trace

`torch_npu.profiler` 使用：

```text
ProfilerLevel.Level1
AiCMetrics.PipeUtilization
analyse_flag=True
async_mode=False
```

因此脚本结束前会同步解析 profile，并检查解析结果是否包含
`DsaSparseLookupUpdate`。[SRC-13]

非零 miss 的 trace 会同时包含状态恢复 copy。分析算子本体时，应在
`kernel_details.csv`、`operator_details.csv` 或 `op_statistic.csv` 中筛选：

```text
DsaSparseLookupUpdate
dsa_sparse_lookup_update
aclnnDsaSparseLookupUpdate
```

### 19.5 建议的对比矩阵

为了区分 all-hit fast path、maintain 成本和 request 并行度，建议至少测试：

| 维度 | 建议值 |
| --- | --- |
| miss rate | `0%`、`1%`、`10%`、`100%` |
| requests | `1`、物理 AIV 数、`2 × AIV 数` |
| profile | Event latency + parsed kernel details |

预期趋势：

- 0 miss：不执行 10K maintain scan；
- 任意非零 miss：进入完整 victim candidate scan；
- request 数不超过 AIV 数时可并行分布到不同 AIV；
- request 数超过 AIV 数时每个 AIV 会处理多个 request。

这只是基于源码控制流给出的趋势预期，具体幅度必须由 A5 profile 验证。

## 20. 当前主要成本与后续观察点

### 20.1 all-hit 路径

主要成本：

- 2K query/mask GM load；
- 2K `index` 随机 lookup；
- protected UB atomic OR；
- 一次 block scan；
- 2K output store。

需要从编译/profile 确认：

- 每线程本地数组是否 spill；
- `index` 随机读取的 DCache/L2 命中率；
- 原子 OR 竞争程度；
- 256-thread occupancy；
- Event latency 中的 device stream gap。

### 20.2 miss 路径

额外成本：

- `index` 上的 atomic CAS；
- canonical miss 分配和双向表写入；
- 10K occupied/protected candidate scan；
- 第二次 block scan；
- 候选区间二次扫描；
- 双向表删除和 free list refill。

只要 `total_misses > 0`，当前代码都会执行完整 10K candidate count scan。因此低比例
miss 下，固定扫描成本可能成为主要瓶颈。

### 20.3 当前没有实现的优化

源码当前没有显式使用：

- `asc_ldcg`/`asc_stcg` cache hint；
- victim hierarchical summary；
- 增量 occupied bitmap；
- exact LRU timestamp；
- maintain 与 I/O overlap；
- graph-specific kernel variant；
- query count 动态 tiling。

这些可以作为后续优化方向，但不能据此否定当前实现；需要根据 profile 中的实际
stall、带宽和 pipeline 指标决定优先级。

## 21. 正确性和验证边界

当前仓库包含：

- 确定性 CPU reference；
- CPU unit tests；
- kernel source structural tests；
- standalone NPU correctness script；
- standalone benchmark；
- `torch_npu.profiler` profile script。[SRC-11][SRC-12][SRC-13]

验证结论应分层表述：

| 验证 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| CPU reference tests | 接口语义、状态不变量、唯一 miss、淘汰顺序 | A5 kernel 正确性 |
| source structural tests | 关键实现结构仍存在 | 设备运行结果 |
| NPU correctness | 当前输入下 NPU 与 CPU oracle 一致 | 所有并发/长期 workload |
| standalone benchmark | metadata op 单调用性能 | 完整模型性能 |
| parsed profiler | kernel/算子级设备指标 | 完整 Indexer/I/O/SFA 性能 |
| PD E2E | 当前测试配置下框架闭环 | 非 mock 的所有生产能力 |

尤其需要保留以下边界：

- metadata op 不等于 KV I/O；
- mock I/O 跑通不等于真实 payload 正确；
- all-hit benchmark 不等于 miss/maintain 性能；
- eager 跑通不等于 graph 跑通；
- 源码推断不等于 A5 实测。

## 22. 源码阅读顺序

建议按以下顺序阅读：

1. [SRC-5]：固定常量和 tiling data。
2. [SRC-8]、[SRC-15]、[SRC-16]：框架如何分配持久状态、构造 lookup 输入并在
   leader/follower layer 间复用结果。
3. [SRC-1]：Torch ABI。
4. [SRC-2]：输入检查和 ACLNN 调用。
5. [SRC-3]：OpDef 与 Ascend 950 注册。
6. [SRC-4]：Host shape 校验、AIV 数量和 workspace。
7. [SRC-6]：AIV kernel shell 和 UB scratch。
8. [SRC-7]：完整 SIMT 算法。
9. [SRC-11]：CPU reference，帮助理解预期语义。
10. [SRC-13]：profile 具体测量边界。

## 23. 参考资料

### 23.1 固定源码引用

[SRC-1]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/torch_binding.cpp#L2991-L3006

[SRC-2]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/attention/dsa_sparse_lookup_update/dsa_sparse_lookup_update_torch_adpt.h#L11-L100

[SRC-3]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/attention/dsa_sparse_lookup_update/op_host/dsa_sparse_lookup_update_def.cpp#L11-L51

[SRC-4]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/attention/dsa_sparse_lookup_update/op_host/dsa_sparse_lookup_update_tiling.cpp#L159-L293

[SRC-5]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/attention/dsa_sparse_lookup_update/op_kernel/dsa_sparse_lookup_update_common.h#L11-L44

[SRC-6]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/attention/dsa_sparse_lookup_update/op_kernel/dsa_sparse_lookup_update.cpp#L11-L64

[SRC-7]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/csrc/attention/dsa_sparse_lookup_update/op_kernel/arch35/dsa_sparse_lookup_update_simt.h#L18-L555

[SRC-8]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/vllm_ascend/attention/dsa_sparse.py#L170-L275

[SRC-9]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/vllm_ascend/ops/dsa_sparse.py#L23-L114

[SRC-10]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/vllm_ascend/attention/dsa_sparse_io.py#L100-L205

[SRC-11]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/tests/ut/ops/dsa_sparse_lookup_update_reference.py#L15-L208

[SRC-12]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/tests/ut/ops/test_dsa_sparse_lookup_update_kernel_source.py#L36-L144

[SRC-13]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/tools/dsa_sparse_lookup_update/profile_operator.py#L42-L340

[SRC-14]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/tools/dsa_sparse_lookup_update/common.py#L241-L321

[SRC-15]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/vllm_ascend/attention/dsa_sparse.py#L634-L768

[SRC-16]: https://github.com/ykfnxx/vllm-ascend/blob/6eb6df91065b7d21cd43bd6c7759f5dd18884326/vllm_ascend/attention/dsa_sparse.py#L883-L913

### 23.2 华为官方 Ascend C / CANN 文档

[CANN-1]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_10840.html

[CANN-2]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10052.html

[CANN-3]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10392.html

[CANN-4]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_10382.html

[CANN-5]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10384.html

[CANN-6]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_10847.html

[CANN-7]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/opdevgapi/atlasophostapi_07_0067.html

[CANN-8]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0171.html

以上 CANN 引用均来自华为昇腾社区官方文档。文档版本覆盖 CANN 8.x、9.0 和
9.1 beta3，主要用于解释跨版本稳定的 API 语义。针对部署环境 CANN 9.1.0 的最终
编译约束，应同时以该环境实际安装的头文件和编译器诊断为准。
