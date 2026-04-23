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

# Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving

> arXiv:2601.21351  
> Chendong Song, Meixuan Wang, Hang Zhou, Hong Liang, Yuan Lyu, Zixi Chen, Yuwei Fan, Zijie Zhou

---

## 1. 问题定义

**AFD (Attention-FFN Disaggregation)** 是一种新兴的 LLM 解码架构，将 Transformer 层拆分为：

- **Attention 端**：有状态、内存受限（KV cache 读取主导）
- **FFN 端**：无状态、计算密集（大批量下可达 compute-bound）

二者通过 **rA-1F 拓扑** 连接——r 个 Attention 实例共享 1 个 FFN 实例，每步解码需要跨实例通信。

### 核心挑战：r* 最优配比问题

- **r 太小** → FFN 饥饿（缺乏输入），算力浪费
- **r 太大** → Attention 排队阻塞等待 FFN，idle time 上升
- 现有系统完全依赖 **经验搜索**，缺乏理论框架

### 难点：Attention 工作负载的非平稳性

每步解码中：
- **Prefill 负载 Pₖ**：常数（新请求的 prompt 长度），E[Pₖ] = B·μ_P
- **Decode 负载 Dₖ**：持续增长（KV cache 累积），E[Dₖ₊₁] = (1−p)(E[Dₖ]+B)
  - 初始 D₀ = 0
  - 饱和值 B·(1−p)/p = B·μ_D
- **总 token 负载 T = P + D** 随步数单调增长，打破理想的 microbatch pipeline 平衡

---

## 2. 系统模型

### 2.1 四阶段同步流水线（每步解码）

```
Attention (r 个并行) → A→F 通信 → FFN (聚合 rB) → F→A 通信 → 下一步
```

1. r 个 Attention worker 各处理 microbatch B 个请求
2. 所有 Attention 将中间激活传给 FFN
3. FFN 处理聚合批次 rB 个激活
4. FFN 将结果返回给 r 个 Attention worker

### 2.2 延迟模型（线性，基于 Roofline 模型）

| 组件 | 延迟公式 | 瓶颈类型 |
|------|----------|----------|
| Attention | t_A(T) = α_A · T + β_A | 内存受限 (memory-bound) |
| FFN | t_F(rB) = α_F · (rB) + β_F | 计算受限 (compute-bound) |
| Communication | t_C(B) = α_C · B + β_C | 带宽受限 |

其中 α 为斜率系数，β 为固定开销。

### 2.3 周期时间

$$\tau(B; r) = \max\{t_A(T),\ t_C(B),\ t_F(rB)\}$$

每步的 cycle time 由最慢组件决定。

### 2.4 Microbatch Pipelining

理想条件下通信延迟被计算重叠掩盖。但 KV cache 增长导致 Attention 延迟逐步增大而 FFN/通信固定，产生 **pipeline bubble**（Figure 2）。

---

## 3. 概率负载模型

### 3.1 请求长度分布

- **Prefill 长度 P**：有界分布，均值 μ_P；具体分布形式不影响分析
- **Decode 长度 D ~ Geo(p)**：几何分布，p = 每步终止概率，均值 μ_D = (1−p)/p
  - 几何分布的 **无记忆性**：X_b(k) ~ Bernoulli(1−p) 独立于 i_b(k)，使 Markov 分析可行

### 3.2 Continuous Batching

请求完成后立即从队列补充新请求，保持每个 Attention 实例恒定 batch size B。

### 3.3 关键随机变量

| 符号 | 含义 |
|------|------|
| X_b(k) ~ Bernoulli(1−p) | slot b 在第 k 步是否继续 |
| s_b(k) | slot b 当前请求的 prefill 长度 |
| i_b(k) | slot b 当前请求已生成的 decode token 数 |
| D_k = Σ_b i_b(k) | 第 k 步聚合 decode 负载 |
| P_k = Σ_b s_b(k) | 第 k 步聚合 prefill 负载 |
| T_k = P_k + D_k | 第 k 步总 token 负载 |

### 3.4 状态更新方程

$$i_b(k+1) = X_b(k) \cdot (i_b(k) + 1) \quad \text{(Eq. 7)}$$

- 若继续 (X=1)：decode index +1
- 若终止 (X=0)：重置为 0

$$s_b(k+1) = X_b(k) \cdot s_b(k) + (1 - X_b(k)) \cdot S'_b(k) \quad \text{(Eq. 8)}$$

- 若继续：保持原 prefill 长度
- 若终止：从新请求重新采样 S'_b(k)

---

## 4. 理论推导

### Lemma 4.1（期望 Token 负载）

$$\mathbb{E}[P_k] = B \mu_P, \quad \forall k \geq 0$$

Prefill 期望负载恒定。

$$\mathbb{E}[D_k] = B \cdot \frac{1-p}{p} \cdot \left(1 - (1-p)^k\right), \quad \forall k \geq 0$$

Decode 期望负载从零出发，以指数形式趋近饱和值 B·μ_D。

**证明思路**：
- Prefill：对 Eq.8 取期望，E[S'_b(k)] = μ_P，得到不动点 Bμ_P
- Decode：对 Eq.7 取期望，E[D_{k+1}|D_k] = (1−p)(D_k + B)，解线性递推

### Definition 4.2（Horizon-Average Token 负载）

$$\bar{T}(B; N) := \frac{1}{K(B)} \sum_{k=0}^{K(B)-1} \mathbb{E}[T_k]$$

其中 K(B) = N/(Bp) 是服务 N 个请求所需的期望解码步数。

### Proposition 4.3（大数定律极限）

$$\bar{T} = \lim_{N \to \infty} \bar{T}(B; N) = \left(\mu_P + \frac{1-p}{p}\right) \cdot B \quad \text{(Eq. 14)}$$

**证明**：将 Lemma 4.1 代入求和公式，对几何级数求和取极限。

**物理意义**：将时变 token 负载用单一稳态值 T̄ 近似，是整个理论的关键简化。

---

## 5. 三区间分析与最优 A/F 比

定义 M(B) := max{t̄_A, t̄_C}（非 FFN 主导延迟）。

### Regime I：Attention 瓶颈 (t̄_A ≥ t̄_C, t̄_A ≥ t̄_F(r))

- 条件：r ≤ r_A，其中

$$r_A := \frac{\bar{t}_A - \beta_F}{\alpha_F B} = \frac{\alpha_A \bar{T} + \beta_A - \beta_F}{\alpha_F B} \quad \text{(Eq. 15)}$$

- Cycle time 锁定在 t̄_A
- 吞吐 per instance = (B/t̄_A) · r/(r+1)，随 r 严格递增
- 最优：r*_I = r_A（边界处取得）

### Regime II：Communication 瓶颈 (t̄_C ≥ t̄_A, t̄_C ≥ t̄_F(r))

- 条件：r ≤ r_C，其中

$$r_C := \frac{\bar{t}_C - \beta_F}{\alpha_F B} \quad \text{(Eq. 16)}$$

- 同理最优：r*_II = r_C

### Regime III：FFN 瓶颈 (t̄_F(r) ≥ M(B))

- 条件：r ≥ r_crit，其中

$$r_{crit} := \frac{M(B) - \beta_F}{\alpha_F B} \quad \text{(Eq. 17)}$$

- 每实例吞吐函数：

$$f(r) = \frac{rB}{(r+1)(\alpha_F r B + \beta_F)}$$

- 对 r 求导令 f'(r)=0，无约束极值点：

$$r_{peak} = \sqrt{\frac{\beta_F}{\alpha_F B}} \quad \text{(Eq. 18)}$$

- f(r) 单峰：r < r_peak 递增，r > r_peak 递减
- 约束最优：r*_III = max{r_crit, r_peak}

---

## 6. 主定理：闭式最优 A/F 比

### Theorem 4.4

$$r^* = \max\left\{\frac{\alpha_A \bar{T} + \beta_A - \beta_F}{\alpha_F B},\quad \frac{\bar{t}_C - \beta_F}{\alpha_F B},\quad \sqrt{\frac{\beta_F}{\alpha_F B}}\right\} \quad \text{(Eq. 19)}$$

最优吞吐：

$$\text{Throughput}^* = \frac{r^* B}{(r^*+1)(\alpha_F r^* B + \beta_F)} \quad \text{(Eq. 20)}$$

### 三项的物理解释

| 项 | 含义 | 场景 |
|----|------|------|
| r_A | Attention-FFN 平衡点 | 低于此值浪费 FFN 算力 |
| r_C | 通信约束 | 实际中通常不是主导 |
| r_peak | FFN 瓶颈区吞吐峰值 | 聚合收益 vs. 拥塞的 tradeoff |

### 实用计算步骤

1. 计算 T̄ ≈ B(μ_P + μ_D)
2. 分别计算三个候选比值 r_A, r_C, r_peak
3. 取 r* = max{r_A, r_C, r_peak}

---

## 7. 实验验证

### 7.1 仿真器设置

- **离散事件仿真器**，6 状态有限状态机：
  ```
  Attention → A2F transfer → Waiting → FFN → F2A transfer → Waiting → repeat
  ```
- 两个 batch 同时运行实现计算重叠
- 连续批处理 (FCFS)：空位立即从缓冲区补充

### 7.2 模型与硬件配置

| 参数 | 值 |
|------|-----|
| 模型 | DeepSeek-V3 |
| Hidden size H | 7168 |
| KV cache 维度 (d_c + d_rope) | 576 |
| Expert 中间维度 | 2048 |
| Expert 总数 | 256 |
| 每 token 路由 expert 数 | k = 8 |
| 硬件 | Huawei Ascend 910C NPU |
| 基准 batch size B | 256 |
| 基准 decode 长度 μ_D | 500 |
| 基准 prefill 长度 μ_P | 100 |
| 仿真规模 N | 10,000 requests/Attention instance |
| 扫描范围 | r ∈ {1, 2, 4, 8, 16, 24, 32} |

### 7.3 延迟系数（Table 2，线性回归拟合执行 trace）

| 参数 | 值 |
|------|-----|
| α_A | 0.00165 cycles/token |
| β_A | 50 cycles |
| α_F | 0.083 cycles/request |
| β_F | 100 cycles |
| α_C | 0.022 cycles/token |
| β_C | 20 cycles |

### 7.4 主要结果

**理论 vs. 仿真最优（Figure 4）**：
- 理论预测 r*_theory ≈ 9.3
- 仿真最优在 r ≈ 8–9 处取得峰值
- **相对误差在 10% 以内**

**Idle 比率分析（Figure 5）**：

| r | FFN idle (η_F) | Attention idle (η_A) |
|---|----------------|---------------------|
| 1 | >60% | ≈10% |
| 8 | ≈η_A（交叉点，均衡配置） | ≈η_F |
| 32 | 接近饱和 | >60% |

**大 r 时系统性偏差**：r=32 时仿真吞吐低于理论约 **15%**，原因是异构 Attention 实例间的 straggler 效应。

### 7.5 消融实验

**Batch Size 影响（Figure 6）**：

| B | r* | 趋势 |
|---|-----|------|
| 128 | 7.08 | |
| 256 | 9.34 | |
| 512 | 10.31 | |

结论：更大 batch size 摊薄固定开销、提高峰值吞吐，r* 随 B 温和增长。

**Workload 分布影响（Figure 7）**：

| 配置 | r* |
|------|-----|
| μ_P=100, μ_D=100 | 2.17 |
| μ_P=100, μ_D=500 | 9.30 |
| μ_P=500, μ_D=500 | 17.25 |

结论：r* 随总上下文长度线性缩放；更长上下文降低峰值吞吐。

---

## 8. 关键假设与简化

1. **线性延迟模型**：Attention ∝ token 数，FFN ∝ batch size，Communication ∝ 数据量
2. **Horizon 平均**：用 N→∞ 极限下的稳态值 T̄ 替代时变 token 负载
3. **几何 decode 分布**：无记忆性使闭式分析可行；已通过生产 trace 验证（Figure 3）
4. **Continuous batching**：假设完美的 slot 补充，未考虑队列饥饿
5. **负载均衡**：假设 r 个 Attention 实例间 token 负载均衡；大 r 时 straggler 效应导致约 15% gap

---

## 9. 结论与未来工作

- 建立了严格的概率框架，尽管 Attention 负载非平稳仍能准确建模 AFD 动态
- 闭式最优 A/F 比公式在多种配置下有效
- 仿真验证理论预测在 ~10% 相对误差内
- **未来工作**：真实系统验证（待 AFD 实现成熟后）、负载均衡策略缓解大 r 时的 straggler gap
