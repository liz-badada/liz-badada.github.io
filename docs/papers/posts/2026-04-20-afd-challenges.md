---
date: 2026-04-20
categories:
  - parallelism
tags: [afd, moe, parallelism, roofline]
arxiv: "2602.09721"
venue: "arxiv"
tier: L4
status: read
---

<!-- more -->

# AFD 真的通用吗？标准集群 + 细粒度 MoE 下的 dead zone

> arXiv:2602.09721v1 · Guowei Liu, Hongming Li, Yaning Guo, Yongxi Lyu, Mo Zhou, Yi Liu, Zhaogeng Li, Yanpeng Wang (Baige AI Team, Baidu Inc.)

## TL;DR {: .tldr-heading }

AFD（Attention-FFN Disaggregation）的宣传卖点是通过极大聚合 FFN 批量在 MoE 模型上榨出高 HFU。但在当前主流的**标准集群**（非 Superpod，比如 H800）+ **细粒度 MoE 模型**（DeepSeek-V3、Kimi-K2、Qwen3-Coder）下，AFD 实际收益远不如宣称。

本文用扩展 roofline + 不均衡敏感性分析给出一份清单式判决：H800 上 DeepSeek-V3 的 AFD 理论 HFU 上限**只有 33%**，且存在 "dead zone"——到了这个上限后**增加 FFN 节点数无法提升 HFU**，因为 scale-out 带宽已经封顶。对 DP/EP 负载不均衡，AFD 的容忍度也显著低于 Large-Scale EP。**Superpod（GB200/GB300）+ 粗粒度 MoE（M ≥ 5120，稀疏度 ≤ 16）** 才是 AFD 的真正用武之地。

行业趋势与 AFD 友好方向相反——MoE 越来越细粒度、越来越稀疏。所以标准集群上，**Large-Scale EP 仍是最佳选择**，AFD 不是通用优化。

## AFD 的性能预算：3BO + SLO

AFD 要用 **3-batch overlap（3BO）**才能隐藏 Attention↔FFN 的 per-layer 通信延迟：

```
Batch 1: Attn  →  A2F  →  FFN  →  F2A
Batch 2:        Attn  →  A2F  →  FFN  →  F2A
Batch 3:              Attn  →  A2F  →  FFN  →  F2A
```

端到端延迟预算方程：

$$T = \text{SLO} \times L_{\text{accept}} = t_g + N_{\text{layers}} \times N_{BO} \times t_B$$

- `T`：单 batch 端到端延迟
- `SLO`：每 token 输出目标时延（TPOT SLA）
- `L_accept`：MTP 平均接受长度
- `t_g`：inter-batch gap（CUDA Graph 启动、边界同步等）
- `N_layers`：层数
- `N_BO = 3`：overlap 度
- `t_B`：每层每 batch 允许的 wall time 预算

3BO 无气泡的硬约束：

$$\max(t_a, t_f, t_c) \le t_B$$

$$2 t_a \ge t_f + t_c, \quad 2 t_f \ge t_a + t_c$$

`t_a, t_f, t_c` 分别是 Attention、FFN、通信的 per-layer 延迟，最优运行点是 `t_a, t_f → t_B`。

## 三层利用率：S_t · OFU · HFU

| 指标 | 定义 | 物理意义 |
|---|---|---|
| Temporal Sparsity `S_t` | `S_t = t_G / t_B` | 算子活跃时间占预算的比例 |
| Operator FLOPS Utilization `OFU` | `OFU = FLOPs / t_G` | 算子自身算力利用率 |
| Hardware FLOPS Utilization `HFU` | `HFU = FLOPs / t_B = OFU × S_t` | 硬件整体算力利用率 |

这个拆分是后面所有分析的基础。**AFD 的 FFN 预算 `t_B` 被 SLO 死死限制**——即使 OFU 很高，若 `S_t` 低（算子没占满预算），HFU 仍然上不去。

## Budget-Based HFU 与 dead zone

MoE stage 的算术强度：

$$I = \frac{2 B_{\text{rank}}}{\lceil N_{\text{experts}} / (N_F \cdot g) \rceil}$$

- `B_rank`：每个 FFN rank 收到的 token 数
- `N_experts`：总专家数
- `N_F`：FFN 节点数
- `g`：组内专家聚合因子

带宽约束下的 `B_rank`：

$$B_{\text{rank}} = \min\left(B_{\text{ScaleOut}} \cdot \max\left(1, \frac{\text{TopK}}{N_F}\right),\ B_{\text{ScaleUp}}\right)$$

`B_ScaleOut` 是跨节点（比如 H800 RDMA）单 rank 带宽，`B_ScaleUp` 是节点内（比如 NVLink）单 rank 带宽。随 `N_F` 增长系统会经过四个区间：scale-up bound（NVLink 饱和）→ stable intensity（算术强度随 N_F 增长）→ **scale-out bound（dead zone）**→ maximum intensity。

**Dead zone** 的形式化表达：当系统进入 scale-out bound，

$$\text{HFU} = \frac{2 \cdot B_{\text{ScaleUp}} \cdot M}{\text{FLOPS}}$$

`M` 是 expert intermediate size。这个 HFU 只取决于 **M 和硬件 FLOPS 之比**，和 `N_F` 无关。**增加 FFN 节点数无法提升 HFU**。这就是论文指出的 H800 上 DeepSeek-V3 HFU 被封顶在 ~33% 的根本原因。

## 不均衡敏感性：AFD 的阿喀琉斯之踵

**DP 不均衡**。起因是 DP ranks 间 context length 不均，某些 rank 的 batch 只有 `σ·B`（σ ≤ 1）。

| 架构 | 吞吐恢复系数 |
|---|---|
| Large-Scale EP | `α_EP > σ`（可通过增大 batch 回收 FFN 释放的时延） |
| AFD | `α_AFD = σ`（固定预算 t_B 不允许 batch 回填） |

AFD **完全承担 σ 的损失**，无法像 EP 那样部分恢复。

**EP 不均衡**。起因是 MoE 路由不均，某些 expert 的 token 数变化。

Large-Scale EP 的惩罚：

$$\alpha_{EP} = \frac{\lambda_{EP} + 1}{\lambda_{EP} + 1/\sigma} > \sigma$$

`λ_EP = t_a / t_f`，通常在 [2, 4]。

AFD 连续近似：

$$\alpha_{\text{exact}} = \frac{\lambda_{AFD} + 1}{\lambda_{AFD} + 1/\sigma} > \sigma$$

`λ_AFD = N_A / N_F`。但 AFD 的实际扩容是节点级离散的，多一个 quantization 损失：

$$\alpha_{\text{floor}} = \frac{(\lambda_{AFD} + 1) \lfloor \sigma N_A \rfloor}{\lfloor \sigma N_A \rfloor + N_F}$$

$$\alpha_{\text{ceil}} = \frac{(\lambda_{AFD} + 1) \lceil \sigma N_A \rceil}{\lceil \sigma N_A \rceil + N_F} \cdot \frac{\sigma N_A}{\lceil \sigma N_A \rceil}$$

即使 FFN 激进优化（`λ = 5`），**AFD 要到 σ = 0.8 才勉强追上 EP**，大多数现实不均衡水平下 AFD 劣于 EP。

## 硬件 × 模型矩阵

假设 `L_accept = 1.7, t_g = 15 ms`，各 GPU × 模型组合的理论 HFU 上界（%）：

| GPU | DeepSeek-V3 | Kimi-K2 | Step3 | Qwen3-Coder | ERNIE-4.5 | GLM-4.7 |
|---|---|---|---|---|---|---|
| H20 | 48 | 48 | 58 | 52 | 54 | 44 |
| H100 | HBM-limited | HBM-limited | 34 | 37 | 36 | 30 |
| H200 | HBM-limited | HBM-limited | 34 | 37 | 36 | 30 |
| **H800** | **33** | **33** | 40 | 36 | 38 | 31 |
| B200 | 36 | 36 | 44 | 39 | 41 | 34 |
| B300 | 36 | 36 | 44 | 39 | 41 | 34 |
| **GB200** | **65.5** | **65.5** | **65.5** | 64 | 65 | 60 |
| **GB300** | **65.5** | **65.5** | **65.5** | 64 | 65 | 60 |

表 2-1 AFD 理论 HFU 上界的硬件 × 模型矩阵
{: .figcaption }

几个结论：

- **标准集群（H800）+ DeepSeek-V3**：AFD 理论 HFU 上限 33.1%，**低于 Large-Scale EP 可达的 ~60%**
- **Superpod（GB200/GB300）**：HFU 上限 65.5%，AFD 才真正有优势
- **H100/H200 + DeepSeek-V3/Kimi-K2**："HBM-limited"，Attention 侧 KV cache 读取就已瓶颈，AFD 无意义
- Step3（M = 5120，粗粒度）天然比 DeepSeek-V3（M = 2048，细粒度）HFU 高

## AFD 友好 vs. 不友好

**模型层面**：

| 条件 | AFD 友好 | AFD 不友好 |
|---|---|---|
| Expert 中间维度 M | 大（Step3: 5120） | 小（DeepSeek-V3: 2048） |
| Expert 稀疏度 N_experts/TopK | 低（Step3: 48/3=16） | 高（DeepSeek-V3: 256/8=32） |
| 路由方式 | 粗粒度 | 细粒度 |

**行业悖论**：当前 MoE 演进趋势恰恰是**更细粒度、更稀疏**（DeepSeek、Kimi、Qwen3），**和 AFD 友好方向相反**。

**硬件层面**：

| 条件 | AFD 友好 | AFD 不友好 |
|---|---|---|
| 拓扑 | Superpod 全互联（GB200/GB300） | 标准 NVLink + RDMA 集群（H800） |
| Scale-up 带宽 | 充裕（> 600 GB/s） | 有限（NVLink 节点内） |
| 子节点划分 | 灵活（缓解离散扩容惩罚） | 刚性（节点级最小粒度） |
| 异构资源 | 可按 Attn/FFN 差异分配 | 同构集群无收益 |

## AFD vs. 其他架构

**AFD vs. Large-Scale EP**：

| 维度 | AFD | Large-Scale EP |
|---|---|---|
| 扩容粒度 | 节点级离散（quantization 损失） | batch 连续可调 |
| 不均衡恢复 | `α = σ`（完全承担） | `α > σ`（部分恢复） |
| 标准集群 HFU 上限 | ~33% (H800/DeepSeek-V3) | ~60% |
| 代码复杂度 | 极高（3BO + per-layer 通信 + 紧同步） | 中 |

**标准集群上 Large-Scale EP 是最佳选择**。

**AFD vs. PD Disaggregation**：

| 维度 | PD | AFD |
|---|---|---|
| 分离对象 | Prefill / Decode 阶段 | Attention / FFN 模块 |
| 耦合程度 | 松耦合（独立实例） | 紧耦合（每层同步） |
| 弹性 | 灵活 | 受 3BO 约束，很脆弱 |
| Jitter 传播 | 阶段间隔离 | 跨 Attn/FFN 迅速扩散 |

**Expert-as-a-Service (EaaS)** 则在小模型（单节点放得下）下更有希望，万亿级 MoE（DeepSeek-V3）受限。

## 实现层面的挑战

即使选择了 AFD 友好的硬件-模型组合，工程上还是苦：

- **CUDA Graph 启动开销**：3BO 下算子数 × 1.5，边界同步成本不可忽略
- **小 t_B 预算**：boundary overhead 占比变大
- **Layout 转换**：需要额外 shuffle 算子实现 Attn ↔ FFN 的布局互换
- **紧同步**：Attn 与 FFN 两阶段必须严格对齐，任何 jitter 快速放大

## AFD Challenges 小结

- AFD 的理论收益**严重依赖硬件-模型组合**，不是通用优化
- 标准集群 + 细粒度 MoE（当前主流）下，AFD HFU 被 scale-out 带宽封顶在 ~33%
- AFD 对 DP/EP 不均衡的容忍度显著低于 Large-Scale EP
- Superpod + 粗粒度 MoE 才是 AFD 用武之地；行业趋势与之相反

给不同角色的建议：

| 角色 | 建议 |
|---|---|
| 推理框架开发者 | 标准集群上继续投资 Large-Scale EP；AFD 路径谨慎评估 ROI |
| 基础设施规划 | 规划 AFD 必须投资 Superpod 级拓扑（GB200/GB300） |
| 模型设计 | 想未来上 AFD，考虑粗粒度专家（与当前细粒度趋势相反） |
| 异构部署 | AFD 的价值在异构资源场景：Attention 需大 HBM、FFN 需大算力时可精准分配 |

判断 AFD 是否值得的四步决策树：

1. 硬件是否 Superpod 级？否 → 放弃 AFD，选 Large-Scale EP
2. 模型是否粗粒度 MoE（M ≥ 5120，稀疏度 ≤ 16）？否 → AFD 收益低
3. 负载是否均衡（σ > 0.95）？否 → AFD 损失远大于 EP
4. 是否追求异构资源匹配？是 → AFD 可考虑

## 相关链接

- [arXiv:2602.09721](https://arxiv.org/abs/2602.09721)

KB 内：

- [AFD Optimal Ratio 闭式解](2026-04-09-afd-optimal-ratio.md)
- [Frontier 仿真器](2026-04-09-frontier-simulator.md)
- [Parallelism →](../../parallelism/index.md)
- [MoE Systems →](../../moe/index.md)
