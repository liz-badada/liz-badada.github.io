---
date: 2026-04-09
categories:
  - parallelism
tags: [afd, parallelism, perf-modeling]
arxiv: "2601.21351"
venue: "arxiv"
tier: L4
status: read
---

<!-- more -->

# AFD 的 r* 最优配比有闭式解吗？

> arXiv:2601.21351 · Chendong Song, Meixuan Wang, Hang Zhou, Hong Liang, Yuan Lyu, Zixi Chen, Yuwei Fan, Zijie Zhou

## TL;DR {: .tldr-heading }

AFD（Attention-FFN Disaggregation）把 Transformer 层拆成 Attention 池和 FFN 池，用 rA-1F 拓扑连接——r 个 Attention 实例共享 1 个 FFN 实例。现有系统靠经验搜索去调 r，没有理论框架。

难点是 **Attention 的工作负载非平稳**：KV cache 按步累积，Decode 负载随解码步数指数式饱和，理想的 microbatch pipeline 平衡被打破。本文给出一个概率负载模型，借几何分布的无记忆性把时变 token 负载用单一稳态值 `T̄` 近似，然后用线性 roofline 延迟模型推出**闭式 r\* 最优解**：

$$r^* = \max\left\{\frac{\alpha_A \bar{T} + \beta_A - \beta_F}{\alpha_F B},\ \frac{\bar{t}_C - \beta_F}{\alpha_F B},\ \sqrt{\frac{\beta_F}{\alpha_F B}}\right\}$$

三项分别对应 Attention 瓶颈、通信瓶颈、FFN 瓶颈三种运行区间。仿真下理论预测的 r\* ≈ 9.3，实测峰值在 r ∈ {8, 9}，**相对误差 < 10%**。大 r 时实测略低于理论（约 15%）是 Attention 实例间 straggler 效应。

自然的问题：这个闭式解在真实系统下还成立吗？Decode 分布非几何（trace 显示有尾）时精度掉多少？

## AFD 和 r* 配比问题

AFD 把 Transformer decoder 按模块切两半：

- **Attention 池**：有状态，memory-bound（KV cache 读取主导）
- **FFN 池**：无状态，compute-bound（大批量可达 compute 峰值）

两侧通过 **rA-1F 拓扑**连接：r 个 Attention 实例共享 1 个 FFN 实例，每步解码跨实例通信。r 决定两端的平衡：

- r 太小 → FFN 饥饿，算力浪费
- r 太大 → Attention 排队等 FFN，idle 上升

现有工业系统全靠 workload-specific 的经验搜索，没有理论指导。

## 四阶段流水线与延迟模型

每步解码是一个同步四阶段流水：

```
Attention (r 个并行) → A→F 通信 → FFN (聚合 rB 请求) → F→A 通信 → 下一步
```

延迟模型用线性 roofline（每个阶段 = 斜率 × 负载 + 固定开销）：

| 组件 | 延迟 | 瓶颈 |
|---|---|---|
| Attention | `t_A(T) = α_A · T + β_A` | memory-bound |
| FFN | `t_F(rB) = α_F · (rB) + β_F` | compute-bound |
| Communication | `t_C(B) = α_C · B + β_C` | 带宽受限 |

周期时间取最慢组件：

$$\tau(B; r) = \max\{t_A(T),\ t_C(B),\ t_F(rB)\}$$

理想条件下通信延迟可被计算重叠掩盖。但 KV cache 单调增长导致 Attention 延迟逐步增大、FFN 延迟固定，产生 **pipeline bubble**。

## 非平稳 Attention 负载怎么分析？

关键简化来自一个假设：**每个请求 Decode 长度服从几何分布 Geo(p)**，`p` 是每步终止概率。几何分布的无记忆性让 Markov 分析可行——`X_b(k) ~ Bernoulli(1-p)` 独立于 `i_b(k)`。

每个 slot `b` 在第 `k` 步有两个量：

- `s_b(k)` = 当前请求的 prefill 长度
- `i_b(k)` = 当前请求已生成的 decode token 数

状态更新：

$$i_b(k+1) = X_b(k) \cdot (i_b(k) + 1)$$
$$s_b(k+1) = X_b(k) \cdot s_b(k) + (1 - X_b(k)) \cdot S'_b(k)$$

继续（X=1）时 decode 计数 +1；终止（X=0）时从新请求采样新长度。对这两个递推取期望，可得（Lemma 4.1）：

$$\mathbb{E}[P_k] = B \mu_P \quad\text{（恒定）}$$

$$\mathbb{E}[D_k] = B \cdot \frac{1-p}{p} \cdot (1 - (1-p)^k)$$

Decode 期望负载从零指数饱和到 `B(1-p)/p = B μ_D`。

## 稳态 token 负载 T̄

把 Lemma 4.1 代入 horizon-average 求和并取 `N → ∞` 极限（Prop. 4.3）：

$$\bar{T} = \left(\mu_P + \frac{1-p}{p}\right) \cdot B$$

这个简单公式是整个理论的支点：**时变 token 负载用单一稳态值 T̄ 近似**。剩下的分析就是对 `τ` 做 max 分解。

## 三区间分析与闭式 r*

定义 `M(B) := max{t̄_A, t̄_C}`（非 FFN 主导延迟）。系统按 r 大小分三个区间。

**Regime I：Attention 瓶颈**（`r ≤ r_A`）。Cycle time 锁定在 `t̄_A`，吞吐 per instance = `(B/t̄_A) · r/(r+1)` 随 r 递增。最优在边界 `r_A`：

$$r_A = \frac{\alpha_A \bar{T} + \beta_A - \beta_F}{\alpha_F B}$$

**Regime II：Communication 瓶颈**（`r ≤ r_C`）。同理 `r_C = (\bar{t}_C - \beta_F) / (α_F B)`。

**Regime III：FFN 瓶颈**（`r ≥ r_crit`）。每实例吞吐：

$$f(r) = \frac{rB}{(r+1)(α_F r B + β_F)}$$

单峰，求导令 `f'(r) = 0` 得无约束极值：

$$r_{\text{peak}} = \sqrt{\frac{β_F}{α_F B}}$$

约束下最优 `r^*_{III} = \max\{r_{\text{crit}}, r_{\text{peak}}\}`。

**主定理**把三个区间合起来：

$$r^* = \max\{r_A,\ r_C,\ r_{\text{peak}}\}$$

三项的物理含义：

| 项 | 物理含义 |
|---|---|
| `r_A` | Attention-FFN 平衡点，低于它 FFN 算力被浪费 |
| `r_C` | 通信约束，实际中通常不是主导 |
| `r_peak` | FFN 瓶颈区的吞吐峰值（聚合收益 vs. 拥塞 tradeoff） |

实用计算只要三步：算 `T̄ ≈ B(μ_P + μ_D)`；算三个候选；取最大。

## 仿真验证

**仿真器**是一个 6 状态有限状态机的离散事件模拟器，两个 batch 并发跑实现计算重叠，FCFS 连续批处理。

**配置**（Huawei Ascend 910C NPU, DeepSeek-V3）：

| 参数 | 值 |
|---|---|
| Hidden H | 7168 |
| KV cache dim (d_c + d_rope) | 576 |
| Expert 中间维度 | 2048 |
| Expert 总数 / Top-K | 256 / 8 |
| 基准 batch size B | 256 |
| 基准 decode 长度 μ_D | 500 |
| 基准 prefill 长度 μ_P | 100 |
| 仿真规模 N | 10,000 请求/实例 |
| r 扫描 | {1, 2, 4, 8, 16, 24, 32} |

延迟系数（线性回归拟合 trace）：`α_A = 0.00165, β_A = 50, α_F = 0.083, β_F = 100, α_C = 0.022, β_C = 20`（单位按论文）。

**主要结果**：理论预测 `r*_theory ≈ 9.3`，仿真最优在 `r ∈ {8, 9}`，相对误差 < 10%。

Idle 比率随 r 变化：

| r | FFN idle | Attention idle |
|---|---|---|
| 1 | > 60% | ≈ 10% |
| 8 | 交叉点（均衡） | 交叉点 |
| 32 | 接近饱和 | > 60% |

r = 32 时仿真吞吐比理论低约 15%，来源是**异构 Attention 实例间的 straggler 效应**。

**Batch size 消融**：

| B | r* |
|---|---|
| 128 | 7.08 |
| 256 | 9.34 |
| 512 | 10.31 |

B 越大 → 摊薄固定开销 → r* 温和增长。

**Workload 分布消融**（r* 随总上下文长度线性缩放）：

| (μ_P, μ_D) | r* |
|---|---|
| (100, 100) | 2.17 |
| (100, 500) | 9.30 |
| (500, 500) | 17.25 |

## 关键假设

这套闭式解漂亮，但依赖五条假设。写下来以后用的时候知道什么时候会失效：

- **线性延迟模型**——Attention 和 token 数、FFN 和 batch size、通信和数据量都是线性关系。非线性区（例如 kernel tile quantization）不成立。
- **Horizon 平均**——用 `N → ∞` 稳态值 `T̄` 替代时变负载。短 session 下 Decode 还没饱和时偏差会大。
- **几何 decode 分布**——论文已用生产 trace 验证；但长尾 workload（比如 reasoning model）可能偏离几何。
- **完美 continuous batching**——slot 一释放立即补，未考虑队列饥饿。
- **负载均衡**——r 个 Attention 实例间 token 均衡；大 r 时 straggler 导致 ~15% gap。

## AFD Optimal Ratio 小结

- 用几何分布的无记忆性 + 线性 roofline 可以把非平稳 AFD 问题化简成闭式 max 分解
- r* 的三个候选（Attention 平衡点、通信约束、FFN 峰值）都有直观物理含义
- 10% 的理论-实测 gap 说明在合理的仿真精度下这个公式是可用的工程工具
- 大 r straggler 是下一步值得攻的方向

后续有真实系统验证或者非几何分布的扩展再回来更新。

## 相关链接

- [arXiv:2601.21351](https://arxiv.org/abs/2601.21351)

KB 内：

- [AFD Challenges（硬件-模型组合下的 dead zone）](2026-04-20-afd-challenges.md)
- [Frontier 仿真器](2026-04-09-frontier-simulator.md)
- [Parallelism →](../../parallelism/index.md)
