---
date: 2026-04-09
categories:
  - perf-modeling
tags: [simulation, moe, disaggregation, perf-modeling]
arxiv: "2508.03148"
venue: "arxiv"
tier: L3
status: read
---

<!-- more -->

# Frontier：为什么现有仿真器撑不起分离式推理？

> arXiv:2508.03148 · Yicheng Feng, Xin Tan, Kin Hang Sew, Yimin Jiang, Yibo Zhu, Hong Xu

## TL;DR {: .tldr-heading }

现代 LLM 推理已经从"单体密集模型"走到 **MoE + 分离式架构**（PD 分离、AF 分离、Expert Parallelism）。但主流仿真器（Vidur、LLMServingSim）是为上一代同址密集模型设计的，三处硬伤：Attention 算子用"序列长度平方根"的代理建模，倾斜 batch 上误差 > 55%；MoE 核心算子 **GroupedGEMM 完全不支持**；replica-centric 抽象表达不了跨集群的 KV-cache 传输、背压、同步原语。

Frontier 的解法是**从 replica-centric 转到 stage-centric**：GlobalController 有状态地编排请求跨集群流转、ClusterWorker 抽象专用硬件集群、ExecutionPredictor 把算子拆成数据依赖的事件序列；Attention 和 GroupedGEMM 都用 Random Forest 回归模型基于分布特征预测，算子级 94–95% 的样本误差在 10% 内。端到端 PD 分离验证下整体相对误差 19–23%。

自然的问题是：算子级做到了，端到端为什么还有 20% 的误差？分离式 workflow 的多阶段 pipeline 误差累积是主因。Frontier 是 replica-centric 过时后的一次架构换代，但"高保真度"目标还没达到。

## 现有仿真器缺了什么

优化一个 72B 模型在 16 GPU 上的配置，经验搜索要 **18,000 GPU-hours（成本 > \$93,000）**。仿真驱动的配置探索是必要的。但现有仿真器在三个维度有根本缺陷。

**Intra-node 建模精度不足**。Vidur 的 Attention 模型用**单一代理长度**（通常是序列长度的平方根）简化建模。在序列长度倾斜的 batch 上，Vidur 的 FlashAttention 预测**误差超过 55%**（预测 0.151 ms vs 实测 0.340 ms）。而且 Vidur **完全不支持 GroupedGEMM**，也就是 MoE 的核心算子。

**Inter-node 编排能力缺失**。旧仿真器用 **replica-centric 抽象**——每个 replica 是同构、自包含的模型副本。无法表达分离式架构里的跨集群路由、KV-cache 传输、复杂同步原语。用 Frontier 作者的话："推理不再是单体任务，而是跨专用异构集群编排的多阶段 workflow。"

**系统级策略缺失**。真实推理引擎（vLLM、SGLang、TRT-LLM）实现了多样的 batching、调度、内存管理策略，现有仿真器把这些关键行为抽象掉了。

## 分离式架构的三种范式

Frontier 要仿真的目标系统有三类：

**PD 分离（Prefill/Decode）**把 compute-bound 的 Prefill 和 memory-bound 的 Decode 阶段分到独立专用集群，用两阶段截然不同的计算特征优化资源分配。

**AF 分离（Attention/FFN）**更细粒度：Attention 和 FFN 跨集群部署，支持更精细的异构扩展。

**MoE + Expert Parallelism**把 expert FFN 分布到专用集群，每个 token 路由到其中一个子集，参数量大幅增长、计算量亚线性增长。

三者都违反了 replica-centric 的基本假设。

## Frontier 架构：Stage-Centric

核心理念一句话：**从"管理同构 replica 池"转到"编排请求在分布式系统中的流转"**。

三层层级：

```
GlobalController
  ├── ClusterWorker (Prefill / Decode / Attention / FFN)
  │     ├── ClusterScheduler
  │     └── ReplicaWorker Pool
  │           └── ExecutionPredictor
  └── ClusterWorker ...
```

图 1-1 Frontier 三层架构：GlobalController 做跨集群编排，ClusterWorker 抽象专用硬件集群，ReplicaWorker 内的 ExecutionPredictor 把算子拆成事件序列
{: .figcaption }

### GlobalController

**有状态编排器**，管理跨分离系统的端到端请求生命周期——协调 ClusterWorker 事件、管理跨阶段 workflow、实现**背压机制**做速率匹配、维护请求状态机。

PD 模式下的 pull-based KV-cache 传输：

```
请求到达
  ↓
GlobalController 路由 → Prefill 集群
  ↓
Prefill 完成：状态 = PREFILL_COMPLETE
KV-Cache 保持在 prefill 内存缓冲区
  ↓
Decode 集群监控内存利用率
  ↓
Decode slot 释放（KV-Cache 驱逐）
Decode ClusterScheduler 发信号通知内存可用 → GlobalController
  ↓
GlobalController 出队 PREFILL_COMPLETE 请求
发起 KV_CACHE_TRANSFER 事件
  ↓
KV-Cache 传输到 Decode 集群
Decode 开始处理
```

关键动态：Decode GPU 有限内存是瓶颈资源；Prefill→Decode 输出速率受 Decode 内存可用性约束；系统级背压防止内存溢出。

AF 模式下 GlobalController 发起 decode step 后，Decode-Attention 把全局 batch 拆成 m 个 micro-batch，动态构建跨 L 层所有操作的**依赖图**。事件驱动引擎按依赖满足顺序调度事件。Ping-pong 流水线延迟隐藏——`A_TO_F_TRANSFER(i, k)` 执行时，Attention GPU 空闲，可执行 `ATTN_COMPUTE(i+1, k)`。总 token 延迟 = 最后一个 micro-batch 最后一层完成的时间戳。

### ClusterWorker & ExecutionPredictor

ClusterWorker 是专用硬件集群（Prefill/Decode/Attention/FFN）的抽象。内部是 ClusterScheduler 管理本地资源和跨阶段协调、ReplicaWorker Pool 持有个体模型实例集合。调度职责包括资源跟踪、内存可用性通知、micro-batch 交接、本地调度策略执行。

**ExecutionPredictor** 是核心创新：**把逻辑层分解为数据依赖的微 workflow 事件序列**，而不是把算子当成黑箱。

```
传统：Layer(inputs) → output
Frontier：Event₁ → Event₂ → ... → Eventₙ（显式依赖、异构执行时间）
```

## 算子建模：用 Random Forest 代替代理变量

**Attention 算子**。Vidur 的问题是忽略 kernel tiling 下的输入异构性（wave quantization）。Frontier 用 Random Forest 回归，输入特征包括序列长度的聚合统计（mean/min/max）、分布特征（variance/skew）、batch size。训练数据来自异构 batch 组合的执行 trace。结果：**> 94% 样本误差 < 10%**，对比 Vidur 55%。

**GroupedGEMM 算子**（MoE 核心）。Vidur 不支持。Frontier 的输入特征：每个 expert 的 token 计数、expert 数量、模型维度、expert 选择率、负载均衡指标（variance / coefficient of variation）。**> 95% 样本误差 < 6%**。

关键点是特征里**显式包含了 expert 负载分布**，即使在高度可变的工作负载下也能做鲁棒预测。

## MoE 仿真：显式建模 Straggler

ExecutionPredictor 把 MoE 层拆成微 workflow：

```
1. Router 门控计算 → GEMM for gating network
2. Token 分配 → 可插拔路由模块生成 token-to-expert 映射
3. Expert 计算（异构）→ 查询 GroupedGEMM 模型取每个 expert 执行时间
4. 同步屏障 → MoE_latency = max(T_expert₁, ..., T_expertₙ)
```

公式：

$$\text{MoE\_latency} = \max(T_{\text{expert}_1}, T_{\text{expert}_2}, \ldots, T_{\text{expert}_N})$$

取最大值**天然捕获了负载不均衡导致的 straggler 效应**——这是 Frontier 和朴素平均建模的关键区别。

Expert Parallelism 约束：

$$\text{attn\_dp} \times \text{attn\_tp} = \text{moe\_tp} \times \text{moe\_ep}$$

（decode-attention 数据并行度 × 张量并行度 = MoE 张量并行度 × expert 并行度）

## 和既有仿真器的能力对比

| 功能 | LLMServingSim | Vidur | Frontier |
|---|---|---|---|
| PD 分离 | ✗ | ✗ | ✓ |
| AF 分离 | ✗ | ✗ | ✓ |
| Pipeline Parallelism | ✓ | ✓ | ✓ |
| Tensor Parallelism | ✗ | ✗ | ✓ |
| Data Parallelism | ✗ | – | ✓ |
| Expert Parallelism | ✗ | – | ✓ |
| Advanced Scheduling | ✓ | ✓ | ✓ |

表 1-1 Frontier 是第一个原生支持 PD/AF 分离 + 完整 parallelism 组合的仿真器
{: .figcaption }

## 实验评估

**硬件**：8-GPU node，NVIDIA A800-SXM4-80GB，400 GB/s NVLink。**软件**：PyTorch 2.3、CUDA 12.1、Ray 2.42.1、FlashInfer 0.1.6、vLLM 0.10.1（SharedStorageConnector KV interface）。**模型**：Qwen2-7B-Instruct。

### 算子级精度

| 算子 | Vidur | Frontier |
|---|---|---|
| Attention | 特定 case 误差 > 55% | > 94% 样本误差 < 10% |
| GroupedGEMM | 不支持 | > 95% 样本误差 < 6% |

Vidur 在 72 请求倾斜序列长度 batch 的 Attention 预测是 0.151 ms，实测 0.340 ms。Frontier 在同类倾斜 batch 上把精度压到 10% 以内。

### 端到端验证（PD 分离，1:1 Prefill:Decode）

| Batch | Avg Input | Output | Profiled (tok/s/GPU) | Predicted (tok/s/GPU) | 相对误差 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 4 | 32 | 1024 | 111.355 | 90.498 | ~18.7% |
| 8 | 128 | 256 | 131.831 | 109.366 | ~17.1% |
| 16 | 256 | 128 | 151.425 | 127.157 | ~16.0% |
| 32 | 32 | 128 | 313.236 | 240.743 | ~23.2% |

整体端到端相对误差 **19–23%**。算子级建模已经显著优于 Vidur，但端到端的累积误差还在——多阶段 pipeline 里各算子误差相互耦合放大。

## 局限

- 端到端精度维持在 19–23% 误差，尚未达到高保真
- 评估仅限单一模型（Qwen2-7B）和单一硬件（A800）
- AF 分离和 MoE EP 的**端到端**验证结果没展示（仅有算子级）
- 缺少大规模系统设计 case study

## Frontier 小结

- 核心贡献是范式转换：**replica-centric → stage-centric**
- 三大挑战对应三大解法：ML-based 算子建模（Random Forest + 分布特征）、GlobalController + 事件依赖图 + 背压、可插拔模块支持多种真实引擎策略
- 算子级做到了 94–95%，端到端还在 19–23%，误差来源是多阶段 pipeline 的累积与交互
- 未来方向：扩展核心算子建模、量化仿真保真度 vs. 计算成本、大规模 case study、开源

后续如果 Frontier 开源或有扩展到 AF / MoE EP 的端到端验证再回来更新。

## 相关链接

- [arXiv:2508.03148](https://arxiv.org/abs/2508.03148)

KB 内：

- [AFD Optimal Ratio 闭式解](2026-04-09-afd-optimal-ratio.md)
- [AFD Challenges](2026-04-20-afd-challenges.md)
- [Perf Modeling →](../../perf-modeling/index.md)
