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

# Frontier: Simulating the Next Generation of LLM Inference Systems

> arXiv:2508.03148  
> Yicheng Feng, Xin Tan, Kin Hang Sew, Yimin Jiang, Yibo Zhu, Hong Xu

---

## 1. 问题动机

现代 LLM 推理正从单体密集模型转向 **MoE + 分离式架构**，现有仿真器（Vidur、LLMServingSim）设计于同址 (co-located) 密集模型时代，存在三大根本缺陷：

### 1.1 Intra-node 建模精度不足

- Vidur 的 Attention 模型使用**单一代理长度（通常为序列长度的平方根）**简化建模
- 在序列长度倾斜的 batch 上，Vidur 的 FlashAttention 预测**误差超过 55%**（预测 0.151ms vs 实测 0.340ms）
- 未支持 MoE 核心算子 **GroupedGEMM**

### 1.2 Inter-node 编排能力缺失

- 旧仿真器采用 **replica-centric 抽象**：每个 replica 为同构、自包含的模型副本
- 无法表达分离式架构中的：跨集群路由、KV-Cache 传输、复杂同步原语
- "推理不再是单体任务，而是跨专用异构集群编排的多阶段 workflow"

### 1.3 系统级策略缺失

- 真实推理引擎（vLLM, SGLang, TensorRT-LLM）实现了多样的 batching、调度、内存管理策略
- 现有仿真器将这些关键行为因子抽象掉

### 成本背景

优化 72B 模型在 16 GPU 上的配置需经验搜索约 **18,000 GPU-hours（成本超 $93,000）**，凸显仿真驱动探索的必要性。

---

## 2. 背景：推理架构范式

### 2.1 PD 分离（Prefill/Decode Disaggregation）

将 **计算密集的 Prefill** 和 **内存带宽受限的 Decode** 阶段分配到独立专用集群，利用两阶段截然不同的计算特征优化资源分配。

### 2.2 AF 分离（Attention/FFN Disaggregation）

更细粒度的拆分：Attention 计算和 FFN 运算跨集群部署，支持更精细的异构扩展。

### 2.3 MoE + Expert Parallelism (EP)

- MoE 用多个 "expert" FFN 替换密集 FFN，每个 token 仅路由到其中一个子集
- 实现参数量大幅增长、计算量仅亚线性增长
- EP 将 expert 计算分布到专用集群

---

## 3. Frontier 架构设计

### 3.1 核心理念：Stage-Centric 设计

**从 "管理同构 replica 池" 转变为 "编排请求在分布式系统中的流转"。**

三层层级架构：

```
GlobalController
  ├── ClusterWorker (Prefill / Decode / Attention / FFN)
  │     ├── ClusterScheduler
  │     └── ReplicaWorker Pool
  │           └── ExecutionPredictor
  └── ClusterWorker ...
```

### 3.2 GlobalController（全局控制器）

**职责**：有状态编排器，管理跨分离系统的端到端请求生命周期。

- 协调独立 ClusterWorker 之间的事件
- 管理跨阶段 workflow 和依赖关系
- 实现 **背压 (backpressure)** 机制进行速率匹配
- 维护请求状态机

#### PD 模式下的 Pull-Based KV-Cache 传输

```
请求到达
  ↓
GlobalController 路由 → Prefill 集群
  ↓
Prefill 完成：请求状态 = PREFILL_COMPLETE
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

**关键动态**：
- Decode GPU 有限内存为瓶颈资源
- Prefill→Decode 输出速率受 Decode 内存可用性约束
- 系统级背压防止内存溢出

#### AF 模式下的事件依赖图

1. GlobalController 发起 decode step
2. Decode-Attention 阶段将全局 batch 划分为 m 个 micro-batch
3. GlobalController 和 ClusterScheduler 动态构建跨 L 层所有操作的**依赖图**
4. 事件驱动引擎按依赖满足顺序调度事件
5. 总 token 生成时间 = 最终事件完成的时间戳

**Ping-Pong 流水线延迟隐藏**：
- A_TO_F_TRANSFER(i, k) 执行时，Attention GPU 空闲，可执行 ATTN_COMPUTE(i+1, k)
- 最终延迟：Token_latency = timestamp(FFN_COMPUTE(m, L))（最后一个 micro-batch 的最后一层）

### 3.3 ClusterWorker（集群工作器）

专用硬件集群的抽象（Prefill/Decode/Attention/FFN 集群）。

内部组件：
- **ClusterScheduler**：管理本地资源、参与跨阶段协调
- **ReplicaWorker Pool**：个体模型实例集合

调度职责：
- 跟踪资源利用率（GPU 内存、计算容量）
- 向 GlobalController 通信内存可用性
- 管理 AF 流水线中的 micro-batch 交接
- 执行本地调度策略（batching、内存管理）

### 3.4 ReplicaWorker & ExecutionPredictor

**ExecutionPredictor** 是核心创新：将逻辑层分解为**数据依赖的微 workflow 事件序列**，而非将算子视为单体黑箱。

```
传统：Layer(inputs) → output
Frontier：Event₁ → Event₂ → ... → Eventₙ（显式依赖、异构执行时间）
```

---

## 4. 算子建模

### 4.1 Attention 算子

**问题**：Vidur 用单一代理长度简化，忽略了 kernel tiling 下的输入异构性（wave quantization 现象）。

**Frontier 方案**：
- **模型类型**：Random Forest 回归
- **输入特征**：
  - 序列长度的聚合统计（mean, min, max）
  - 分布特征（variance, skew）
  - Batch size
  - 序列长度分布特性
- **训练数据**：来自异构 batch 组合的执行 trace
- **精度**：>94% 的预测误差低于 10%（vs. Vidur 55% 误差）

### 4.2 GroupedGEMM 算子（MoE 核心）

Vidur 完全不支持，是性能预测的"重大盲区"。

**Frontier 方案**：
- **输入特征**：
  - 每个 expert 的 token 计数
  - Expert 数量
  - 模型维度
  - Expert 选择率
  - 负载均衡指标（token 分布的 variance / coefficient of variation）
- **精度**：>95% 的预测误差低于 6%

### 4.3 训练方法论

特征"反映输入属性和 expert 负载分布"，即使在高度可变的工作负载下也能实现"鲁棒且精确的预测"。

---

## 5. MoE 推理仿真

### 5.1 Token-to-Expert 路由与 Straggler 建模

ExecutionPredictor 将 MoE 层拆解为显式微 workflow：

```
1. Router 门控计算 → GEMM for gating network
2. Token 分配 → 可插拔路由模块生成 token-to-expert 映射
3. Expert 计算（异构）→ 查询 GroupedGEMM 模型获取每个 expert 的执行时间
4. 同步屏障 → MoE_latency = max(T_expert₁, T_expert₂, ..., T_expertₙ)
```

**关键公式**：

$$\text{MoE\_latency} = \max(T_{\text{expert}_1}, T_{\text{expert}_2}, \ldots, T_{\text{expert}_N})$$

取所有 expert 执行时间的最大值，天然捕获了负载不均衡导致的 **straggler 效应**。

### 5.2 Expert Parallelism 约束

虚拟模型分片遵循拓扑约束：

$$\text{attn\_dp} \times \text{attn\_tp} = \text{moe\_tp} \times \text{moe\_ep}$$

（decode-attention 数据并行度 × 张量并行度 = MoE 张量并行度 × expert 并行度）

### 5.3 负载不均衡处理

不预测平均 expert 时间，而是显式模拟"由 token 负载不均衡导致的最坏情况 straggler"，确保"不均衡效应以高保真度建模"。

---

## 6. 功能对比（Table 1）

| 功能维度 | LLMServingSim | Vidur | **Frontier** |
|---------|:---:|:---:|:---:|
| PD 分离 | ✗ | ✗ | **✓** |
| AF 分离 | ✗ | ✗ | **✓** |
| Pipeline Parallelism | ✓ | ✓ | **✓** |
| Tensor Parallelism | ✗ | ✗ | **✓** |
| Data Parallelism | ✗ | – | **✓** |
| Expert Parallelism | ✗ | – | **✓** |
| Advanced Scheduling | ✓ | ✓ | **✓** |

---

## 7. 实验评估

### 7.1 实验配置

| 项目 | 配置 |
|------|------|
| 硬件 | 8-GPU node, NVIDIA A800-SXM4-80GB |
| 互联 | 400 GB/s NVLink |
| 软件栈 | PyTorch 2.3, CUDA 12.1, Ray 2.42.1, FlashInfer 0.1.6 |
| 推理引擎 | vLLM 0.10.1 (SharedStorageConnector KV interface) |
| 测试模型 | Qwen2-7B-Instruct |

### 7.2 算子级精度

**Attention 算子（Figure 2）**：
- Frontier：>94% 的 case 误差 < 10%
- Vidur：特定 case 误差超 55%（72 请求倾斜序列长度 batch：0.151ms vs 0.340ms）

**GroupedGEMM 算子**：
- Vidur：不支持
- Frontier：>95% 的误差 < 6%

### 7.3 端到端验证（PD 分离，1:1 Prefill:Decode 比）

| Batch | Avg Input | Output | Profiled (tok/s/GPU) | Predicted (tok/s/GPU) | 相对误差 |
|:-----:|:---------:|:------:|:--------------------:|:---------------------:|:--------:|
| 4 | 32 | 1024 | 111.355 | 90.498 | ~18.7% |
| 8 | 128 | 256 | 131.831 | 109.366 | ~17.1% |
| 16 | 256 | 128 | 151.425 | 127.157 | ~16.0% |
| 32 | 32 | 128 | 313.236 | 240.743 | ~23.2% |

**整体端到端误差范围：19.0%–23.2%**

### 7.4 结果分析

- 算子级建模显著优于 Vidur，特别是在输入异构性高的场景
- 端到端验证表明趋势预测准确，但仍存在 19–23% 的误差裕量
- 误差来源：多阶段 pipeline 中各算子误差的累积与交互

---

## 8. 局限性

1. 端到端精度维持 19–23% 误差，尚未达到高保真
2. 评估仅限单一模型 (Qwen2-7B) 和单一硬件配置 (A800)
3. AF 分离和 MoE EP 的端到端验证结果未展示（仅有算子级）
4. 缺少大规模系统设计 case study

---

## 9. 结论与未来工作

**核心贡献**：从 "replica-centric" 到 **"stage-centric"** 的仿真架构范式转换，实现了对分布式多阶段推理 workflow 的原生建模。

三大贡献对应三大挑战：
1. **Intra-node**：ML-based 细粒度算子建模（Random Forest + 分布特征）
2. **Inter-node**：GlobalController + 事件依赖图 + 背压机制
3. **系统策略**：可插拔模块支持多种真实推理引擎策略

**未来方向**：
- 扩展核心算子建模
- 量化仿真保真度与计算成本
- 大规模系统设计 case study
- 开源发布
