# vLLM / vLLM-Ascend HiSparse 社区调研

> 调研时间：2026-07-23
>
> 调研范围：SGLang HiSparse、vLLM 稀疏注意力与 KV offload、vLLM-Ascend 相关基础能力
>
> 目标：判断 vLLM 社区是否已有与 SGLang HiSparse 等价的实现，并为 ASU-Ascend 后续设计提供基线

## 1. 结论

截至 2026-07-23，vLLM 社区已经出现与 SGLang HiSparse 高度相似、甚至明确声明为
HiSparse port 的实现，但尚未合入 vLLM `main`：

1. [vLLM PR #46326](https://github.com/vllm-project/vllm/pull/46326)
   是当前最完整的公开实现。它将完整 Sparse MLA KV 放在 pinned host memory，
   GPU 只保留 Indexer cache 和每请求 hot buffer，并按 Indexer TopK 结果执行
   LRU lookup、miss swap-in 和 slot remap。
2. [vLLM RFC #48203](https://github.com/vllm-project/vllm/issues/48203)
   是更活跃、更通用的社区方案。它把 prefill layerwise offload 与 decode sparse
   offload 合并，并讨论如何接入 HMA、KVConnector、P/D 和现有 Sparse MLA backend。
3. [vLLM RFC #33980](https://github.com/vllm-project/vllm/issues/33980)
   代表较早的 sparse KV offload 路线。其最新回复还报告了一种 CUDA VMM
   host-mapped KV 实现，但尚未提交公开 PR。
4. vLLM `main` 已分别具备 Sparse MLA/DSA 和通用 CPU KV offload 的基础能力，
   但二者尚未组成 HiSparse 所需的逐层、逐 token TopK miss-load 路径。
5. vLLM-Ascend 已合入 SFA MLA KV 与 Indexer cache 的独立分配能力，但截至调研时，
   官方仓库还没有公开的 HiSparse、DSA offload 或 KVIO DSA PR。

因此，当前状态应表述为：

> vLLM 社区已有可审查的 HiSparse 类实现和活跃设计，但 upstream `main`
> 尚无可直接启用的正式功能；ASU-Ascend 可以参考其数据面与状态机，
> 不能把它当作已经稳定的上游接口。

## 2. 判断“类似 HiSparse”的标准

SGLang HiSparse 的核心不是普通 CPU KV cache，也不只是 Sparse Attention kernel。
判断一个方案是否与 HiSparse 等价，需要同时满足以下数据流：

```text
完整 Indexer cache 驻留设备
        |
        v
Indexer 生成每层 TopK token
        |
        v
查询每请求/每层 GPU hot buffer
        |
        +-- hit  -> 复用 resident slot
        |
        +-- miss -> 从 host full MLA KV 加载到 GPU hot buffer
                         |
                         v
              TopK token remap 为 hot-buffer slot
                         |
                         v
                    Sparse MLA / SFA
```

完整历史 MLA KV 必须继续保留在 host，不能通过丢弃或压缩历史 KV 换取容量。
GPU 的 MLA 占用由 hot-buffer budget 决定，而不是随完整上下文长度线性增长。

这与 vLLM 已有通用 offload 的主要区别是：

| 路径 | 搬运粒度 | 搬运触发条件 | 主要目标 |
| --- | --- | --- | --- |
| 通用 CPU offload | KV/prefix block | prefix 命中和 block 生命周期 | 扩大二级缓存容量 |
| HiSparse | TopK miss row | 当前 step 的 Indexer TopK | 降低 Sparse MLA 常驻 HBM |

因此，仅有 CPU KV offload 或仅有 Sparse MLA backend 都不能视为完整 HiSparse。

## 3. SGLang HiSparse 基线

[SGLang PR #20343](https://github.com/sgl-project/sglang/pull/20343)
于 2026-03-23 合入，是本次比较的基线。其目标是将 decode 期间不活跃的 MLA KV
保存在 CPU memory，仅把稀疏注意力当前需要的 KV 保留或换入 GPU，以提高并发
batch size 和吞吐。

后续与本次调研最相关的增强包括：

- [SGLang PR #21591](https://github.com/sgl-project/sglang/pull/21591)：
  P/D 模式下将 Prefill KV 直接传输到 Decode DRAM，避免先完整落入 Decode HBM。
- [SGLang PR #21932](https://github.com/sgl-project/sglang/pull/21932)：
  调整 decode 新 token 备份时机，减少 overlap scheduling 中的 CPU bubble。

这些 PR 说明 HiSparse 不是一个孤立 kernel，而是同时涉及：

- host/device 两级 KV pool；
- request admission 和 per-request device-buffer 预算；
- decode 新 token 的 host backup；
- TopK hit/miss 与 LRU 状态；
- P/D KV 传输目的地址；
- CUDA graph、调度和资源释放。

## 4. vLLM 已经合入的基础能力

### 4.1 Sparse MLA / DSA

vLLM 已经具备完整的模型原生稀疏注意力基础：

- [PR #25896](https://github.com/vllm-project/vllm/pull/25896)：
  合入 DeepSeek-V3.2 模型和 DSA 运行路径。
- [PR #26441](https://github.com/vllm-project/vllm/pull/26441)：
  注册 `FLASHMLA_SPARSE` backend。
- [PR #37735](https://github.com/vllm-project/vllm/pull/37735)：
  为 DSA 模型增加 IndexCache 支持。

这些能力解决了：

```text
完整 Indexer K -> TopK selection -> Sparse MLA attention
```

但默认 MLA KV 仍按正常 KV cache 方式驻留 GPU，没有 TopK 驱动的 host miss-load。

### 4.2 CPU KV offload

vLLM 也已合入两条主要 CPU KV offload 路径：

- [PR #22595](https://github.com/vllm-project/vllm/pull/22595)：
  `OffloadingConnector`。
- [PR #37160](https://github.com/vllm-project/vllm/pull/37160)：
  `SimpleCPUOffloadConnector`，复用 `BlockPool` 和 `KVCacheCoordinator`。

它们主要处理完整 block 的二级缓存、prefix reuse 和异步 block copy。
它们不直接接收 Indexer TopK，也不完成以下链路：

```text
TopK -> resident lookup -> miss token load -> physical hot slot -> Sparse MLA
```

这就是 vLLM `main` 与 HiSparse 之间仍然存在的核心缺口。

## 5. vLLM 直接 HiSparse port：PR #46326

[PR #46326](https://github.com/vllm-project/vllm/pull/46326)
标题即为 “HiSparse: host-resident sparse-MLA decode hot-buffering”，并明确说明是
SGLang HiSparse 的 port。

### 5.1 实现结构

该 PR 的核心路径是：

```text
Indexer TopK
  -> hisparse_swap_in
       -> 按 global KV slot 查询每请求 LRU hot buffer
       -> hit 直接复用
       -> miss 从 pinned host pool gather 到 GPU
       -> 更新 LRU 和 token-to-hot-slot 映射
  -> 使用 remapped indices 执行 FlashMLA Sparse
```

与 SGLang 初版相比，该 PR 还声明增加了：

1. hot entry 以 global KV slot 为 key，从而不需要独立的 per-request host-location table；
2. block 复用时主动 invalidation，避免复用旧 slot 映射；
3. `FULL_DECODE_ONLY` CUDA graph 支持；
4. NIXL mixed DRAM/VRAM transfer；
5. GLM-5.2 `index_topk_freq=4` 场景的 plan-once；
6. shared layers 的 gather prefetch 与计算重叠。

### 5.2 当前成熟度

截至调研时，该 PR：

- 状态为 Open、Draft；
- 目标分支是 `releases/v0.24.0`，不是 vLLM `main`；
- 修改 19 个文件，新增约 4,272 行；
- 尚无人工 review；
- 将 vLLM `main` 适配和 MTP 支持列为后续工作。

PR 中的吞吐、准确率、生产占用率和稳定性数据均为提交者报告。本次调研没有复现
CUDA、NIXL、P/D 或 GLM-5.2 运行结果，因此这些数据只能作为方案证据，不能视为
社区已经独立验证的结论。

## 6. vLLM 通用方案：RFC #48203

[RFC #48203](https://github.com/vllm-project/vllm/issues/48203)
提出同时覆盖 prefill 和 decode 的方案。

### 6.1 Prefill

Prefill 使用 layerwise KV offload：

- 只分配固定数量的 device KV layer buffers；
- 当前层 forward 前 onload；
- 当前层结束后 offload，并并行 onload 下一层；
- 通过 2～4 个 device layer buffers 尝试覆盖传输时间。

### 6.2 Decode

Decode 使用与 HiSparse 相同的核心结构：

- 完整 MLA KV 保存在 host；
- Indexer cache 全量驻留 device；
- 每请求/每层分配固定 `n * topk` hot buffer；
- Indexer TopK 后检查 hot buffer；
- 只从 host 加载 miss KV；
- 将 token IDs remap 为 hot-buffer slots 后执行 Sparse Attention。

RFC 作者报告，在 DeepSeek-V3.2 上使用 `2 * topk` device buffer 可获得约
80%～90% cache hit rate。该数字同样属于 RFC 作者实验结果，本次没有独立复现。

### 6.3 社区讨论中的关键问题

维护者整体对方向持正面态度，但仍在讨论：

1. 是否修改现有 Sparse MLA backend，还是增加新的 offload-aware backend；
2. TopK buffer 应放在 HMA 内还是作为固定 runtime buffer 独立分配；
3. block-level H2D 和 token-level H2D 是否应使用不同的传输实现；
4. 是否创建独立 offloading connector；
5. 首期只支持 P/D disaggregation 后，如何扩展到 co-located 和其他 connector；
6. 如何兼容 MTP、CUDA graph、DeepSeek-V4 和 MiniMax-M3 等组合。

RFC 作者表示正在基于 vLLM-Ascend 实现，并计划提交公开 PR。调研时官方
`vllm-project/vllm` 和 `vllm-project/vllm-ascend` 中尚未发现该新 PR。

## 7. 其他相关路线

### 7.1 RFC #33980：TopK preload 与 VMM host-mapped KV

[RFC #33980](https://github.com/vllm-project/vllm/issues/33980)
最早提出：

- 完整 KV offload 到 CPU；
- GPU 使用固定 TopK KV buffer；
- 根据相邻 step/layer TopK 相似性进行 preload；
- actual TopK 产生后只补 miss KV。

该 RFC 一度因 graph mode、传输覆盖和现有功能兼容问题停滞。
2026-07-23 的最新回复报告了另一种已运行实现：

- 用 CUDA VMM 建立连续虚拟地址；
- 前部 block 由 HBM backing，尾部 block 由 host backing；
- Sparse Attention kernel 继续使用 global slot index；
- 不使用 TopK staging buffer，也不执行 stage-back copy；
- 报告已在单机 8×H100、GLM-5.2、1M context 运行。

该实现尚无公开 PR，无法检查代码、测试和实际性能边界。

### 7.2 RFC #37263：hotness-aware multi-level KV

[RFC #37263](https://github.com/vllm-project/vllm/issues/37263)
提出 query-aware block scoring、冷热 block 分层和批量传输。它保留完整 KV，
但选择粒度偏向 block representation，而不是模型 Indexer 的原生 TopK token。

该 RFC 已因长期不活跃被关闭，状态为 `not_planned`，目前不应视为可继续依赖的
upstream 路线。

### 7.3 RFC #48445：SparDA lookahead prefetch

[RFC #48445](https://github.com/vllm-project/vllm/issues/48445)
希望使用独立 Forecast projection 在第 N 层预测第 N+1 层要访问的 KV，
提前通过 KVConnector 发起 prefetch。

它主要解决：

- offloaded KV 的 PCIe latency；
- 当前层得到 TopK 后才开始传输造成的串行等待；
- 稀疏选择本身的复杂度。

它不是完整 HiSparse 实现，更适合作为 sparse offload 的后续预取优化。

### 7.4 PR #49121：实验性衍生分支

[PR #49121](https://github.com/vllm-project/vllm/pull/49121)
标题为 “Exp/hisparse routed experts”。其改动文件包含 PR #46326 的 HiSparse
kernel/backend/config 文件，并额外加入 routed-experts capture。

从改动范围判断，它更像在 HiSparse 分支上叠加 routed-experts 统计的实验版本。
由于 PR 描述、测试结果和人工 review 均不足，不应将其作为独立的正式实现基线。

## 8. vLLM-Ascend 当前基础

[vLLM-Ascend PR #11647](https://github.com/vllm-project/vllm-ascend/pull/11647)
已经合入，完成了 SFA MLA KV 与 Indexer cache 的独立 spec、容量计算、物理 tensor
分配和绑定。

这项改动是 sparse offload 的重要前置条件，因为 HiSparse 类实现必须区分：

```text
Indexer plane：完整上下文，常驻 device
MLA plane：完整历史在 host，device 只保留 hot buffer
```

但 PR #11647 本身不实现：

- host full MLA KV pool；
- resident token/slot lookup；
- TopK miss compaction；
- token-level host-to-device batch load；
- LRU/eviction；
- SFA indices remap；
- P/D host destination handoff。

截至调研时，vLLM-Ascend 官方仓库中没有以 HiSparse、DSA offload 或 KVIO DSA
为主题的公开 PR。RFC #48203 作者声明的 Ascend 实现应继续跟踪，但在 PR 出现前，
不能把它计入已交付功能。

## 9. vLLM-Ascend main 的版本对应关系

vLLM-Ascend `main` 不是固定绑定一个 vLLM release tag。根据官方
[Versioning Policy](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/community/versioning_policy.md)：

- `main` 持续跟踪一个通过 Ascend CI 验证的 vLLM main commit；
- 同时兼容最新 1～2 个 vLLM release；
- 固定版本交付使用 `releases/vX.Y.Z` 分支。

截至 2026-07-23，vLLM-Ascend `main` 记录：

| 锚点 | 当前值 | 含义 |
| --- | --- | --- |
| `.github/vllm-main-verified.commit` | [`54503ecec0f3ac31e5ecfc5f28652e4cc42307b5`](https://github.com/vllm-project/vllm/commit/54503ecec0f3ac31e5ecfc5f28652e4cc42307b5) | main 开发应使用的精确 vLLM commit |
| `.github/vllm-release-tag.commit` | `v0.25.1` | 当前 release 兼容锚点 |

因此，PR #46326 虽然实现完整，但它基于 vLLM `releases/v0.24.0`。若要移植到当前
vLLM-Ascend `main`，必须按 verified vLLM commit 的 KV cache、ModelRunner、
Sparse MLA backend 和 connector 接口重新适配，不能直接把它视为 v0.25.1/main
可用补丁。

## 10. 对 ASU-Ascend 的建议

### 10.1 参考优先级

建议按以下优先级使用社区成果：

1. 用 RFC #48203 作为 upstream 架构与接口讨论基线；
2. 用 PR #46326 作为 CUDA hot-buffer、LRU、slot remap、graph 和 NIXL 的实现参考；
3. 用 vLLM-Ascend PR #11647 作为 Ascend split Indexer/MLA cache 的当前上游基础；
4. 用 RFC #33980 的 VMM 路线作为对照实验，不在缺少 PR 时作为主路径；
5. 不再基于已关闭的 RFC #37263 设计主接口。

### 10.2 需要保持的实际调用链

ASU-Ascend 后续实现应保持以下调用语义：

```text
每层 Indexer TopK
  -> resident lookup
  -> hit/miss 生成
  -> miss token/row compaction
  -> host/KVIO batch load 到目标 hot slots
  -> 更新 token_to_slot / slot_to_token
  -> TopK remap 为 SFA indices
  -> SFA
```

其中必须区分：

- 原始 TopK token position；
- resident logical slot；
- KV cache physical slot；
- host/KVIO storage offset；
- 最终交给 SFA 的 mapped indices。

### 10.3 不应直接继承的假设

1. 不应把普通 block-level OffloadingConnector 当作 token-level sparse load。
2. 不应假设 PR #46326 的 CUDA kernel、NIXL 注册和 graph capture 语义可直接移植到 NPU。
3. 不应使用 Python `.cpu().tolist()` 和逐 descriptor 循环作为最终 graph-safe 热路径。
4. 不应把作者自报吞吐当作 Ascend 性能结论。
5. 不应仅按 vLLM tag 适配 `main`；应使用
   `.github/vllm-main-verified.commit` 的精确上游接口。

## 11. 验证边界

本次完成的是公开源码、PR metadata、Issue 讨论和分支策略调研：

- 已确认相关 PR/Issue 在 2026-07-23 的公开状态；
- 已检查 PR #46326 和 #49121 的目标分支、改动文件和 review 状态；
- 已确认 vLLM-Ascend `main` 的 verified vLLM commit 与 release tag；
- 未运行 SGLang/vLLM HiSparse；
- 未复现提交者报告的 CUDA、NIXL、P/D、1M context 或吞吐数据；
- 未进行 Ascend NPU 编译和运行时验证。

## 12. PR / Issue 汇总

| 仓库 | PR / Issue | 状态（2026-07-23） | 主题 | 与 HiSparse / ASU 的关系 |
| --- | --- | --- | --- | --- |
| SGLang | [PR #20343](https://github.com/sgl-project/sglang/pull/20343) | Merged | **HiSparse 核心实现**：host-resident KV、GPU hot buffer、稀疏 decode | 本次比较的功能基线 |
| SGLang | [PR #21591](https://github.com/sgl-project/sglang/pull/21591) | Merged | **P/D 直接写 Decode DRAM** | 说明 HiSparse 需要 host 目的地址的数据面 |
| SGLang | [PR #21932](https://github.com/sgl-project/sglang/pull/21932) | Merged | **Decode token backup 调度优化** | 说明新 token host backup 属于热路径调度问题 |
| vLLM | [PR #46326](https://github.com/vllm-project/vllm/pull/46326) | Open, Draft | **HiSparse 直接 port**：pinned host Sparse MLA KV、LRU hot buffer、slot remap | 当前最完整公开实现，但基于 `releases/v0.24.0`，尚未合入 |
| vLLM | [Issue #48203](https://github.com/vllm-project/vllm/issues/48203) | Open | **Layerwise prefill + sparse decode KV offload RFC** | 最活跃、最可能形成通用 upstream 方案 |
| vLLM | [Issue #33980](https://github.com/vllm-project/vllm/issues/33980) | Open, stale 后重新活跃 | **Sparse KV offload、TopK preload、VMM host-mapped KV** | 有工作实现报告，但尚无公开 PR |
| vLLM | [Issue #37263](https://github.com/vllm-project/vllm/issues/37263) | Closed, not planned | **Hotness-aware multi-level KV cache** | 概念相似但已停止推进 |
| vLLM | [Issue #48445](https://github.com/vllm-project/vllm/issues/48445) | Open | **SparDA Forecast projection 与 lookahead prefetch** | 属于 sparse offload 的后续预取优化 |
| vLLM | [PR #49121](https://github.com/vllm-project/vllm/pull/49121) | Open | **HiSparse + routed-experts capture 实验** | 可能叠加于 #46326，不宜作为正式基线 |
| vLLM | [PR #25896](https://github.com/vllm-project/vllm/pull/25896) | Merged | **DeepSeek-V3.2 / DSA 模型支持** | 提供模型原生 Indexer 和稀疏注意力基础 |
| vLLM | [PR #26441](https://github.com/vllm-project/vllm/pull/26441) | Merged | **注册 FLASHMLA_SPARSE backend** | 提供 Sparse MLA 执行基础 |
| vLLM | [PR #37735](https://github.com/vllm-project/vllm/pull/37735) | Merged | **DSA IndexCache 支持** | 为分组共享 TopK/Indexer 优化提供基础 |
| vLLM | [PR #22595](https://github.com/vllm-project/vllm/pull/22595) | Merged | **OffloadingConnector** | block-level CPU KV offload 基础，不等价于 TopK miss load |
| vLLM | [PR #37160](https://github.com/vllm-project/vllm/pull/37160) | Merged | **SimpleCPUOffloadConnector** | 复用 BlockPool/HMA 的 CPU offload 基础，不等价于 HiSparse |
| vLLM-Ascend | [PR #11647](https://github.com/vllm-project/vllm-ascend/pull/11647) | Merged | **SFA MLA KV 与 Indexer cache 独立分配** | Ascend sparse offload 的关键前置能力，但尚未实现 host miss-load |
