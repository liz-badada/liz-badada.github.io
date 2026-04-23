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

# Revealing the Challenges of Attention-FFN Disaggregation for Modern MoE Models and Hardware Systems

> arXiv:2602.09721v1
> Guowei Liu, Hongming Li, Yaning Guo, Yongxi Lyu, Mo Zhou, Yi Liu, Zhaogeng Li, Yanpeng Wang (Baige AI Team, Baidu Inc.)

---

## 1. 核心问题

**AFD (Attention-FFN Disaggregation)** 把 Transformer 层的 Attention 与 FFN 分别部署在不同 GPU 池，通过极大聚合 FFN 批量在 MoE 模型上获取高 HFU。但在当前主流的 **标准集群（非 Superpod，如 H800）** 与 **细粒度 MoE 模型** 的组合下，AFD 实际收益远不如宣称。

本文用扩展屋顶线模型 + 不均衡敏感性分析，系统揭示：**AFD 并非通用优化，而是对"硬件-模型"组合高度挑剔**。

### 1.1 三大贡献

| 贡献 | 结论 |
|------|------|
| **Budget-based HFU 上界** | 标准集群存在 "dead zone"：增加 FFN 节点数无法提升 HFU，因工作负载已被 scale-out 带宽封顶 |
| **不均衡敏感性** | AFD 因节点级离散扩容约束，对 DP/EP 负载不均衡的容忍度低于 Large-Scale EP |
| **可行配置综合** | **Superpod + 粗粒度 MoE** 才是 AFD 的最佳场景 |

---

## 2. 背景：3BO 与延迟预算

### 2.1 3-Batch Overlap 流水线

AFD 需用 **3 个 batch 重叠（3BO）** 才能隐藏 Attention↔FFN 的 per-layer 通信延迟：

```
Batch 1: Attn  →  A2F  →  FFN  →  F2A
Batch 2:        Attn  →  A2F  →  FFN  →  F2A
Batch 3:              Attn  →  A2F  →  FFN  →  F2A
```

### 2.2 延迟预算方程

$$T = \text{SLO} \times L_{\text{accept}} = t_g + N_{\text{layers}} \times N_{BO} \times t_B$$

- **T**: 单 batch 端到端延迟
- **SLO**: 每 token 输出目标时延（TPOT SLA）
- **L_accept**: MTP 平均接受长度
- **t_g**: inter-batch gap（CUDA Graph 启动、边界同步等）
- **N_layers**: 层数
- **N_BO = 3**: overlap 度
- **t_B**: 每层每 batch 允许的 wall time 预算

### 2.3 硬约束

要使 3BO 无气泡运行：

$$\max(t_a,\ t_f,\ t_c) \le t_B$$
$$2t_a \ge t_f + t_c, \quad 2t_f \ge t_a + t_c$$
$$t_a, t_f \to t_B \text{（最优占用）}$$

其中 t_a, t_f, t_c 分别是 Attention、FFN、通信的 per-layer 延迟。

---

## 3. 性能指标：三层利用率

| 指标 | 定义 | 物理意义 |
|------|------|----------|
| **Temporal Sparsity S_t** | S_t = t_G / t_B | 算子实际活跃时间占预算的比例 |
| **Operator FLOPS Utilization (OFU)** | OFU = FLOPs / t_G | 算子自身的算力利用率 |
| **Hardware FLOPS Utilization (HFU)** | HFU = FLOPs / t_B = OFU × S_t | 硬件整体的算力利用率 |

**关键洞察**：AFD 的 FFN 预算 t_B 被 SLO 死死限制，即使 OFU 很高，若 S_t 低（即 t_G/t_B 小，算子没占满预算），HFU 仍然上不去。

---

## 4. Budget-Based HFU 分析

### 4.1 算术强度

MoE stage 的 arithmetic intensity：

$$I = \frac{2 B_{\text{rank}}}{\lceil N_{\text{experts}} / (N_F \cdot g) \rceil}$$

- **B_rank**: 每个 FFN rank 收到的 token 数
- **N_experts**: 总专家数
- **N_F**: FFN 节点数
- **g**: 组内专家聚合因子

### 4.2 带宽约束下的 B_rank 上限

$$B_{\text{rank}} = \min\left(B_{\text{ScaleOut}} \cdot \max\left(1,\ \frac{\text{TopK}}{N_F}\right),\ B_{\text{ScaleUp}}\right)$$

- **B_ScaleOut**: 跨节点（scale-out）单 rank 带宽（如 H800 RDMA）
- **B_ScaleUp**: 节点内（scale-up）单 rank 带宽（如 NVLink）

### 4.3 四种运行区间

1. **Scale-up bound**: NVLink 饱和，B_rank 由 B_ScaleUp 决定
2. **Stable intensity**: 算术强度随 N_F 增长
3. **Scale-out bound**: RDMA 饱和，**这就是 "dead zone"**：增加 N_F 无效
4. **Maximum intensity**: 理论极限

### 4.4 Dead Zone：标准集群的 HFU 上界

当系统进入 **scale-out bound** 时：

$$\text{HFU} = \frac{2 \cdot B_{\text{ScaleUp}} \cdot M}{\text{FLOPS}}$$

- **M**: expert intermediate size
- HFU 仅取决于 **M 与硬件 FLOPS 之比**，与 N_F 无关
- 这就是论文指出的 **标准集群 H800 的 HFU 被封顶在 ~33%** 的根本原因

---

## 5. 不均衡敏感性（AFD 的阿喀琉斯之踵）

### 5.1 DP 不均衡

起因：DP ranks 间 context length 不均，导致某些 rank 的 batch 只有 σ·B（σ ≤ 1）。

| 架构 | 吞吐恢复系数 |
|------|-------------|
| **Large-Scale EP** | α_EP > σ（可通过增大 batch 回收 FFN 释放出的时延） |
| **AFD** | α_AFD = σ（固定预算 t_B 不允许 batch 回填） |

**结论**：AFD 完全承担 σ 的损失，无法像 EP 那样部分恢复。

### 5.2 EP 不均衡

起因：MoE 路由不均衡，某些 expert 的 token 数变化。

**Large-Scale EP 惩罚（Eq. 12）**：

$$\alpha_{EP} = \frac{\lambda_{EP} + 1}{\lambda_{EP} + 1/\sigma} > \sigma$$

- λ_EP = t_a / t_f，通常 λ_EP ∈ [2, 4]

**AFD 连续近似（Eq. 13）**：

$$\alpha_{\text{exact}} = \frac{\lambda_{AFD} + 1}{\lambda_{AFD} + 1/\sigma} > \sigma$$

- λ_AFD = N_A / N_F

**AFD 离散扩容的额外损失（Eqs. 14-15）**：

$$\alpha_{\text{floor}} = \frac{(\lambda_{AFD} + 1) \lfloor \sigma N_A \rfloor}{\lfloor \sigma N_A \rfloor + N_F}$$

$$\alpha_{\text{ceil}} = \frac{(\lambda_{AFD} + 1) \lceil \sigma N_A \rceil}{\lceil \sigma N_A \rceil + N_F} \cdot \frac{\sigma N_A}{\lceil \sigma N_A \rceil}$$

### 5.3 关键发现

- 当 σ·N_A 非整数时，离散扩容引入额外 quantization 损失
- 即使 FFN 激进优化（λ = 5），AFD 在 σ = 0.8 处才勉强追上 EP
- **AFD 在大多数现实不均衡水平下劣于 Large-Scale EP**

---

## 6. 硬件 × 模型矩阵：理论 HFU 上界

假设 L_accept = 1.7，t_g = 15 ms。

### 6.1 HFU 上界表（%）

| GPU | DeepSeek-V3 | Kimi-K2 | Step3 | Qwen3-Coder | ERNIE-4.5 | GLM-4.7 |
|-----|-------------|---------|-------|-------------|-----------|---------|
| H20 | 48 | 48 | 58 | 52 | 54 | 44 |
| H100 | HBM-limited | HBM-limited | 34 | 37 | 36 | 30 |
| H200 | HBM-limited | HBM-limited | 34 | 37 | 36 | 30 |
| **H800** | **33** | **33** | 40 | 36 | 38 | 31 |
| B200 | 36 | 36 | 44 | 39 | 41 | 34 |
| B300 | 36 | 36 | 44 | 39 | 41 | 34 |
| **GB200** | **65.5** | **65.5** | **65.5** | 64 | 65 | 60 |
| **GB300** | **65.5** | **65.5** | **65.5** | 64 | 65 | 60 |

### 6.2 对比洞察

- **标准集群（H800）DeepSeek-V3**：AFD 理论 HFU 上限仅 33.1%，**低于 Large-Scale EP 可达的 ~60%**
- **Superpod（GB200/GB300）**：理论 HFU 可达 65.5%，此时 AFD 才有优势
- **H100/H200 + DeepSeek-V3/Kimi-K2**："HBM-limited"，注意力侧 KV cache 读取即已瓶颈，AFD 无意义
- **Step3**（M = 5120，粗粒度）天然 HFU 比 DeepSeek-V3（M = 2048，细粒度）高

---

## 7. AFD 友好配置 vs. 不友好配置

### 7.1 模型层面

| 条件 | 友好 | 不友好 |
|------|------|--------|
| Expert 中间维度 M | 大（Step3: 5120） | 小（DeepSeek-V3: 2048） |
| Expert 稀疏度 N_experts/TopK | 低（Step3: 48/3=16） | 高（DeepSeek-V3: 256/8=32） |
| 路由方式 | 粗粒度 | 细粒度 |

**行业悖论**：当前趋势正是向 **更细粒度、更稀疏** 的 MoE 演进（DeepSeek、Kimi、Qwen3），与 AFD 友好方向相反。

### 7.2 硬件层面

| 条件 | 友好 | 不友好 |
|------|------|--------|
| 拓扑 | Superpod 全互联（GB200/GB300） | 标准 NVLink + RDMA 集群（H800） |
| Scale-up 带宽 | 充裕（>600 GB/s） | 有限（NVLink 节点内） |
| 子节点划分 | 灵活（减轻离散扩容惩罚） | 刚性（节点级最小粒度） |
| 异构资源 | 可按 Attn/FFN 差异分配 | 同构集群无收益 |

---

## 8. AFD vs. 其他架构

### 8.1 AFD vs. Large-Scale EP

| 维度 | AFD | Large-Scale EP |
|------|-----|----------------|
| 扩容粒度 | 节点级离散（quantization 损失） | batch 连续可调 |
| 不均衡恢复 | α = σ（完全承担） | α > σ（部分恢复） |
| 标准集群 HFU 上限 | ~33% (H800/DeepSeek-V3) | ~60% |
| 代码复杂度 | 极高（3BO + per-layer 通信 + 紧同步） | 中 |

**结论**：**标准集群上，Large-Scale EP 仍是最佳选择。**

### 8.2 AFD vs. PD Disaggregation

| 维度 | PD | AFD |
|------|-----|-----|
| 分离的是什么 | Prefill / Decode 阶段 | Attention / FFN 模块 |
| 耦合程度 | 松耦合（独立实例） | 紧耦合（每层同步） |
| 弹性扩缩容 | 灵活 | 受 3BO 约束，很脆弱 |
| Jitter 传播 | 阶段间相对隔离 | 跨 Attention/FFN 迅速扩散 |

### 8.3 AFD vs. Expert-as-a-Service (EaaS)

EaaS 可行性依赖模型规模：小模型（单节点能放下）更有希望；万亿级 MoE（如 DeepSeek-V3）则受限。

---

## 9. 实现挑战

- **CUDA Graph 启动开销**：3BO 下算子数 × 1.5，边界同步成本不可忽略
- **小 t_B 预算**：boundary overhead 占比变大
- **Layout 转换**：需要额外 shuffle 算子实现 Attn ↔ FFN 的数据布局互换
- **紧同步**：Attn 与 FFN 两阶段必须严格对齐，任何 jitter 快速放大

---

## 10. 结论与建议

### 10.1 主要结论

1. **AFD 的理论收益严重依赖于硬件-模型组合**，不是通用优化
2. **标准集群（H800 + DeepSeek-V3 类细粒度 MoE）下，AFD 的 HFU 上界已被 scale-out 带宽封顶**（~33%），无 dead zone 之外的空间
3. **AFD 对 DP / EP 不均衡的容忍度显著低于 Large-Scale EP**

### 10.2 给各角色的建议

| 角色 | 建议 |
|------|------|
| **推理框架开发者** | 标准集群上继续投资 Large-Scale EP；AFD 路径谨慎评估 ROI |
| **基础设施规划** | 若规划 AFD 部署，必须投资 Superpod 级拓扑（GB200/GB300） |
| **模型设计** | 若希望未来上 AFD，考虑 **粗粒度专家**（与当前细粒度趋势相反） |
| **异构部署** | AFD 的价值在于异构资源场景：Attention 需大 HBM，FFN 需大算力时可精准分配 |

### 10.3 AFD 合理性判定流程

1. 硬件是否 Superpod 级？否 → 放弃 AFD，选 Large-Scale EP
2. 模型是否粗粒度 MoE（M ≥ 5120，稀疏度 ≤ 16）？否 → AFD 收益低
3. 负载是否均衡（σ > 0.95）？否 → AFD 损失 >> EP
4. 是否追求异构资源匹配？是 → AFD 可考虑
