# DSA Sparse Hidden-State Next-Cohort Prefetch 设计草案

> 文档状态：草案，冻结已确认的数据流，保留未决实现项
>
> 编写日期：2026-07-30
>
> 目标实现分支：`dsa-sparse-0.23-eager`
>
> 目标分支审计锚点：`74f00dddc7fd76411058acd1d798084c65dc05ef`
>
> 参考原型分支：
> `work/dev_lookup_maintain_integration_prefetch_with_hidden_states`
>
> 参考原型锚点：`939825c2a0671d2e6a75411030e4682a434546c5`

## 1. 目标

在 `dsa-sparse-0.23-eager` 的 cohort、正式 Lookup 状态和逐层 Hot Main
Cache 架构上，引入基于 hidden states 的 next-cohort Top-K 预测，并将预测
结果接入下一 cohort 的正式 Lookup/Update 和 payload prefetch。

目标不是只生成 `predicted_topk`，而是形成以下闭环：

```text
source cohort leader hidden states
    -> predict next cohort Top-K
    -> update target cohort formal lookup/residency state
    -> derive predicted miss and destination slot
    -> prefetch target cohort Hot Main payload
    -> target cohort computes actual Top-K
    -> actual lookup/update reconciles prediction
    -> load remaining misses
    -> existing SFA consumes actual slot resolution
```

本草案只讨论 eager、单-token normal decode。Graph、SpecDecoding、MTP 和
真实 I/O backend 不在首期范围。

## 2. 术语和数据所有权

### 2.1 Cohort

Cohort 是 Top-K 和 Lookup/驻留状态的所有权边界：

```text
cohort
├── 一个 leader layer
├── 零个或多个 follower layer
├── 一份正式 lookup/residency state
├── leader 计算 semantic Top-K
└── followers 复用 leader 的 Lookup 结果
```

`skip_topk=false` 的物理层创建新 cohort 并成为 leader；
`skip_topk=true` 的物理层加入前一个 cohort。

### 2.2 Indexer K Cache

Indexer K Cache 保存 Indexer 用来计算 Top-K 的 K，例如：

```text
model.layers.*.self_attn.indexer.k_cache
```

它属于模型的 Indexer 计算路径，不是 Hot Main Cache，也不是本草案中的
Lookup/驻留元数据。

### 2.3 Cohort lookup/residency state

本草案不再使用含义不明确的“IndexCache state”称呼。正式名称统一为：

```text
cohort lookup/residency state
```

它由以下持久 Tensor 组成：

```text
index
slot_to_index
free_slots
free_head
```

其语义是：

```text
index[request, semantic_position] -> hot_slot
slot_to_index[request, hot_slot]   -> semantic_position
free_slots / free_head            -> slot allocation state
```

它只保存 semantic token 与本地 Hot Cache slot 的映射和分配状态，不保存
MLA KV payload。

### 2.4 Per-layer Hot Main Cache

Hot Main Cache 保存实际 MLA KV payload，并且仍然逐物理层独立：

```text
one cohort lookup/residency state
├── leader layer Hot Main Cache
├── follower layer 1 Hot Main Cache
└── follower layer 2 Hot Main Cache
```

cohort 内各层使用相同的本地 slot 编号，但 slot `N` 在不同物理层的 Hot
Main Cache 中保存各层自己的 payload。

## 3. 已冻结设计决策

### 3.1 预测输入

预测输入固定取 source cohort leader 计算当次真实 Top-K 时使用的同一份
hidden states。

不从 cohort 的最后一个 follower 获取 hidden states，也不允许 follower
单独发起 hidden-state Top-K 预测。

### 3.2 预测目标

只预测紧邻的下一个 cohort：

```text
source cohort i -> target cohort i+1
```

不按 `physical_layer_id + 1` 决定目标，不跨多个 cohort 预测。

当模型没有跨物理层共享 Top-K/Lookup 状态时，每个物理层都是一个单层
cohort，因此：

```text
next cohort == next physical layer
```

这只是 cohort 退化为单层后的表现，框架接口仍保持
`source_cohort -> target_cohort`。

### 3.3 Lookup 执行顺序

当前真实请求必须优先于下一 cohort 的预测预取。全局顺序固定为：

```text
actual lookup_i
    -> predicted lookup_(i+1)
    -> actual lookup_(i+1)
    -> predicted lookup_(i+2)
```

不允许 predicted lookup 抢在当前 cohort 的 actual lookup 前执行。

predicted lookup 可以在 `actual lookup_i` 完成后提交，并与当前 cohort
后续的 payload I/O、SFA、FFN 以及 follower layer 计算重叠。

### 3.4 Predicted lookup 修改正式状态

`predicted lookup_(i+1)` 直接读写 target cohort `i+1` 的正式
lookup/residency state：

```text
index
slot_to_index
free_slots
free_head
```

首期不使用 shadow lookup state，也不建立独立 speculative cache。

错误预测允许占用正式 resident slot。该 slot 后续由正常 Lookup/Update
淘汰机制回收。

### 3.5 Actual lookup 始终执行

target cohort 到达时必须根据 `actual_topk_(i+1)` 再执行一次正式
Lookup/Update。

predicted lookup 的输出只用于提前 payload I/O，不能直接作为 SFA 的最终
attention indices。

第二次 actual lookup 负责：

```text
predicted and actual overlap -> 正常 hit
actual-only position         -> 新 miss 和新 payload load
predicted-only position      -> 继续 resident，等待正常淘汰
```

最终 SFA 寻址只能由 actual lookup 输出和 live-tail 映射共同确定。

## 4. 目标执行链

### 4.1 Source cohort

```text
source cohort leader hidden states
    |
    +-> current Indexer computes actual_topk_i
    |
    +-> actual lookup/update_i
           |
           +-> current slot_out_i / miss_out_i
           |      |
           |      `-> current cohort per-layer payload I/O and SFA
           |
           `-> predict topk_(i+1)
                  |
                  `-> predicted lookup/update_(i+1)
                         |
                         `-> target cohort payload prefetch
```

逻辑优先级必须保证 `actual lookup_i` 先执行。实现可以在其完成后将预测和
预取提交到独立 stream，以便与当前 cohort 后续计算重叠。

### 4.2 Target cohort

```text
target cohort i+1 arrives
    |
    +-> target leader computes actual_topk_(i+1)
    |
    +-> ensure in-flight predicted writes cannot race with slot update
    |
    +-> actual lookup/update_(i+1)
    |      |
    |      +-> reuse predicted hits
    |      +-> resolve prediction misses
    |      `-> produce final slot_out / miss_out
    |
    +-> load actual-only misses
    |
    `-> per-layer existing SFA
```

## 5. Cohort 与逐层 payload 的关系

对于多层 cohort：

```text
cohort A = [layer 0 leader, layer 1 follower, layer 2 follower]
cohort B = [layer 3 leader, layer 4 follower]
```

预测关系固定为：

```text
layer 0 计算 actual_topk_A 使用的 hidden states
    -> predict cohort B leader 的 semantic Top-K
```

不为 layer 1、layer 2 单独预测 Top-K。

target cohort B 的 predicted Lookup 只执行一次，但其 Lookup 结果可能驱动：

```text
layer 3 Hot Main Cache payload prefetch
layer 4 Hot Main Cache payload prefetch
```

究竟首期只预取 leader payload，还是预取整个 target cohort 的逐层 payload，
仍是待决策项。

## 6. 与参考原型分支的差异

参考分支只在 `vllm_ascend/attention/sfa_v1.py` 中加入 hidden-state
预测原型，其主要行为包括：

- 通过 `physical_layer_id + 1` 寻找下一物理层；
- 使用固定 `hidden_states * 0.1 + 1e-5` 近似下一层 hidden states；
- 重复执行下一层 MLA/Indexer 前处理；
- 调用 LightningIndexer 生成 `predicted_topk`；
- 将预测 K 写入下一层真实 MLA/Indexer cache；
- 最终没有将 `predicted_topk` 接入 Lookup、payload I/O 或下一层消费；
- 没有独立 stream、event、completion handle 或 wait 关系。

因此参考分支只提供“hidden states 可以用于预测后续 Top-K”的实验入口，不能
直接作为当前 eager 架构的实现。

迁移时不得直接继承以下实现：

- 模块级 `global_weight_dict`；
- 硬编码 `physical_layer_id + 1`；
- 硬编码 `alpha=0.1`、`beta=1e-5`；
- 预测路径写入下一层真实 MLA/Indexer cache；
- 生成 `predicted_topk` 后丢弃；
- 在当前 stream 上串行重复下一层全部计算。

## 7. 框架适配需求

### 7.1 ModelRunner

需要基于当前 ordered cohort layout 建立显式映射：

```text
source cohort key
source leader layer
target cohort key
target leader layer
target cohort member layers
```

最后一个 cohort 不创建预测目标。

### 7.2 SFA/Indexer 接入点

source cohort leader 需要暴露：

- 本层 actual Top-K；
- 计算该 Top-K 使用的 hidden states；
- `actual lookup_i` 已完成的触发点。

Prefetch 只能在 `actual lookup_i` 之后触发。

### 7.3 Coordinator 与 step state

当前“一 cohort 每 step 只允许一次 lookup”的状态需要拆分为：

```text
prefetch lookup state/output/completion
actual lookup state/output/completion
```

建议至少显式记录：

```text
prefetch_submitted
prefetch_lookup_output
prefetch_io_completion
actual_lookup_complete
actual_lookup_output
```

predicted lookup 不得设置或替代现有 actual `lookup_complete` 语义。

### 7.4 Lookup operator

predicted 和 actual 两次调用使用同一个正式 Lookup/Update 算子接口，并读写
同一份 target cohort lookup/residency state。

算子需要支持同一 request、同一 decode execution 内对同一 target cohort
状态连续调用：

```text
predicted lookup/update
actual lookup/update
```

必须验证重复 token、错误预测、额外 miss 和淘汰后的反向映射一致性。

### 7.5 I/O operator

I/O 接口需要支持区分：

```text
predicted prefetch I/O
actual miss I/O
```

并提供明确 completion 语义。首期仍可使用 mock I/O，但 mock 必须保留
submit、completion、wait/consume 和 abort 的调用拓扑，不能继续只表示一个
无状态同步 no-op。

### 7.6 生命周期

Prefetch 状态必须纳入：

- batch begin；
- cohort step begin/finish；
- request finish；
- preemption/resume；
- abort；
- coordinator poison；
- request index release。

异步 prefetch 未完成时，不得重置或释放其 target cohort/request 状态。

### 7.7 内存管理

需要在固定 HBM 预算中增加：

- predictor scratch；
- predicted Top-K；
- predicted lookup outputs；
- I/O completion/event。

所有长期使用的地址必须在运行前稳定分配，不在 decode 热路径临时扩容。

## 8. 尚未决定的问题

### 8.1 Predictor 形式

需要决定：

- hidden-state 预测公式；
- 参数是逐 source-target cohort pair 还是共享；
- 参数的训练或离线标定方式；
- BF16/FP16/量化支持范围；
- 是否设置置信度阈值；
- predicted Top-K 数量是否固定为 2048。

### 8.2 Target cohort payload 范围

需要选择：

- 只预取 target leader 的 Hot Main payload；
- 预取 target cohort 所有物理层 payload；
- leader 优先，followers 按剩余窗口分批提交。

### 8.3 Predicted I/O 与 actual lookup 的并发安全

predicted lookup 已经把 slot 标记为 resident，但 payload I/O 可能尚未完成。
如果 actual lookup 在旧 I/O 完成前淘汰并复用该 slot，旧 I/O 可能覆盖新的
payload。

首期候选方案是 target cohort 到达后，先等待 predicted I/O 完成，再执行
actual lookup。该规则简单但可能暴露 I/O 尾延迟，尚未冻结。

若不采用整批等待，则需要设计：

- per-slot completion；
- slot generation/version；
- eviction 与 in-flight write 的冲突检测；
- actual lookup 对 in-flight predicted slot 的处理规则。

### 8.4 Stream 和优先级

需要决定：

- 一个全局 prefetch stream 还是每 cohort 一个 stream；
- actual lookup 与 predicted lookup 的 stream/event 依赖；
- 如何保证 actual lookup 拥有更高执行优先级；
- predictor、Lookup 和 payload I/O 是否使用同一预取 stream；
- target cohort 到达时的最小 wait 粒度。

### 8.5 错误预测资源控制

虽然错误预测允许进入正式 resident state，但仍需决定：

- 每 request 的预取 admission 上限；
- 是否限制 predicted miss 数量；
- 是否根据历史 recall 动态关闭预取；
- 错误预测占用 slot 对正式命中率的可接受影响。

## 9. 首期限制

首期建议保持：

- eager only；
- Decode consumer only；
- `AscendAttentionState.DecodeOnly`；
- 每 request 每 model forward 一个 decode token；
- 只预测 next cohort；
- 不支持 SpecDecoding/MTP；
- 不支持 SP/CP；
- Main payload I/O 继续使用 mock；
- 不修改现有 SFA 算子 ABI；
- 不以 CPU reference 或静态检查替代 A5/CANN 运行验证。

## 10. 验证要求

### 10.1 功能和状态一致性

至少覆盖：

- 单层 cohort；
- leader + followers 多层 cohort；
- 最后一个 cohort 不发起预测；
- predicted Top-K 与 actual Top-K 完全一致；
- 部分重合；
- 完全不重合；
- predicted duplicate；
- actual-only miss；
- predicted-only resident；
- predicted false positive 被后续正常淘汰；
- 同一正式状态连续 predicted/actual 两次 Lookup；
- request finish/preempt/abort 时有 in-flight prefetch；
- request index 重用前不存在旧 completion。

### 10.2 预测质量

必须记录：

```text
predicted/actual Top-K overlap
recall
precision
predicted miss count
actual-only miss count
predicted-only resident count
false-positive eviction count
```

### 10.3 性能

必须区分并记录：

```text
predictor latency
predicted lookup latency
predicted payload I/O latency
target cohort wait latency
actual-only miss I/O latency
end-to-end TPOT
```

同时通过 NPU timeline 验证执行顺序确实为：

```text
actual lookup_i
    -> predicted lookup_(i+1)
    -> actual lookup_(i+1)
```

并验证 predicted lookup/payload I/O 与当前 cohort 后续计算存在真实重叠。

## 11. 非目标

本草案不包含：

- 直接移植参考分支代码；
- 预测下一个 scheduler decode step；
- 一次预测多个后续 cohort；
- follower 独立预测 Top-K；
- shadow lookup state；
- 独立 speculative Hot Cache；
- Graph capture/replay；
- SpecDecoding/MTP；
- 真实 I/O backend 实现；
- SFA operator/schema/kernel 修改。
