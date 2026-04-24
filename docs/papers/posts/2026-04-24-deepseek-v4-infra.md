---
date: 2026-04-24
categories:
  - large-scale
tags: [deepseek, infra, ep-overlap, tilelang, fp4-qat, kv-cache, rl-infra, sandbox]
arxiv: ""
venue: "huggingface"
tier: L3
status: read
---

<!-- more -->

# DeepSeek-V4 基础设施拆解：6144 FLOPs/Byte 是怎么来的？

> By: DeepSeek-AI · 2026-04-24 发布
>
> 架构篇：[DeepSeek-V4：MLA 彻底不用了？](2026-04-24-deepseek-v4.md)
>
> 对应 Tech Report Section 3 / 5.1 / 5.2 / 6

## TL;DR {: .tldr-heading }

架构篇（Section 2）讲了 V4 模型层面"换了什么"。这篇讲的是让这些架构跑起来的**工程**——对做 inference/training infra 的人来说，Section 3 和 5.2 比 Section 2 更直接相关。

其中有几个值得单独拎出来的结论：

第一，**V4 单卡 EP 做 comm/compute overlap 的目标点是 `C/B ≤ 2d = 6144 FLOPs/Byte`**（V4-Pro 的具体数值）。Tech Report 直接给 NV / AMD / 华为下战书："bandwidth 堆过这个阈值就不再有边际收益，把硅面积让给算力"。

第二，**FP4 → FP8 dequantization 是无损的**（FP8 E4M3 比 FP4 E2M1 多 2 个 exponent bits 足够吸收 fine-grained scale）。这条对 Hopper 部署很关键——之前架构篇里我对 "Hopper 代差拉大" 的判断要修正，memory-bound 场景下 H100 / H200 吃到 V4 的大部分收益。

第三，**传统 PagedAttention 的一刀切 page 假设被 V4 打破了**。Hybrid Attention 导致每层 KV cache 大小不一、还要和 Sliding Window 的 state cache 分开管。SGLang day-0 PR 走的是 RadixAttention 变种而不是 PagedAttention，就是这个原因。

顺着 Tech Report Section 3 的组织讲，从 EP overlap 一直到 DSec sandbox。末尾接 Section 5 的 post-training 方法学和 Section 6 的官方自评。

## Fine-Grained EP Overlap

### Comm latency 为什么可以被隐藏

V4 做 expert parallelism 的 comm/compute overlap 把 V3 的 DualPipe 思路又细化了一级。起点是 Tech Report 的一个关键 profile 观察：

> "Our profiling reveals that within a single MoE layer, the total time of communication is less than that of the computation. Therefore, after fusing communication and computation into a unified pipeline, computation remains the dominant bottleneck, implying that the system can tolerate lower interconnect bandwidth without degrading end-to-end performance."

每个 MoE layer 有 4 阶段：

```
Dispatch (comm) → Linear-1 (compute) → Activation → Linear-2 (compute) → Combine (comm)
```

总 comm 时间 < 总 compute 时间，所以关键不是"怎么减少 comm"，而是"怎么把 comm 藏进 compute 里"。

### Wave 调度

Comet（Zhang et al. 2025b，对标 baseline）的做法是 Dispatch 和 Linear-1 overlap、Linear-2 和 Combine overlap，两个 coarse-grained 区间。V4 把粒度又降了一级——**把 experts 切成 waves**，每个 wave 是一小撮 experts。稳态下一个 wave 的 comm 完成立刻启动它的 compute，同时下一 wave 的 token transfer、上 wave 的结果回传三路并发。expert-level pipeline 全程连续，compute 和 comm 都不空转。

效果（官方数字，对标 non-fused baseline）：

| 场景 | 加速 |
|---|---|
| 常规推理 workload | 1.50 ~ 1.73× |
| RL rollout / 高速 agent serving（长尾 small batch） | 1.96× |

实现以 **MegaMoE** 的名字开源，作为 DeepGEMM 的一个组件：[github.com/deepseek-ai/DeepGEMM/pull/304](https://github.com/deepseek-ai/DeepGEMM/pull/304)。在 NVIDIA GPU 和华为 Ascend NPU 两个平台都验证过。

### 6144 FLOPs/Byte

Tech Report 3.1 末尾直接给硬件厂商下了一份"客户需求书"，对 NV infra 视角是罕见的甲方输入。核心是推导出 compute-to-communication ratio 的目标点：

设峰值算力 `C`、互联带宽 `B`。全部 overlap 的充要条件是 `C/B ≤ V_comp / V_comm`。对 V4-Pro，每个 token-expert 对需要：

- FLOPs：`6hd`（SwiGLU 的 gate / up / down 三个 projection；`h` = hidden size，`d` = intermediate size）
- Bytes：`3h`（FP8 Dispatch + BF16 Combine）

代入：

$$\frac{C}{B} \leq \frac{V_{\text{comp}}}{V_{\text{comm}}} = \frac{6hd}{3h} = 2d = 6144 \text{ FLOPs/Byte}$$

也就是每 1 GB/s 互联带宽足够吃掉 6.1 TFLOP/s 的算力。DeepSeek 的话讲得很直白：

> "Once bandwidth meets this threshold, it ceases to be the bottleneck, and devoting additional silicon area to further bandwidth brings diminishing returns. We encourage future hardware designs to target such balance points rather than scale bandwidth unconditionally."

除了这个带宽 / 算力比，Tech Report 还顺带提了三条给硬件设计的建议：

**Power budget**。极端 kernel fusion 会让 compute / memory / network 同时满载，power throttling 变成关键瓶颈。下一代硬件需要为 concurrent workload 预留足够的 power headroom。

**Communication primitives**。V4 选了 pull-based ——每个 GPU 主动从远端 GPU 读数据——原因是 fine-grained push 的 notification latency 太高。对 NVSHMEM IBGDA 的 push-based 语义是暗戳戳的批评。未来硬件需要 low-latency cross-GPU signaling 才能让 push 可行。

**Activation function**。Tech Report 直接提议用没有 exp / division 的 element-wise 激活替代 SwiGLU。对下一代模型是个信号——SwiGLU 未必是最终答案，exp 对硬件太不友好。

## TileLang Kernel 开发

V4 几乎所有 custom kernel 都用 [TileLang](https://github.com/tile-ai/tilelang)（Wang et al., ICLR 2026）写，不直接写 CUDA、也不依赖 CUTLASS。Tech Report 给的理由是：

> "In practice, our elaborate model architecture would have resulted in hundreds of fine-grained Torch ATen operators."

V4 架构（CSA、HCA、mHC、Hybrid MoE）要生成的 fused kernel 数量很大，手写 CUDA 不现实。TileLang 是一个 DSL，能快速迭代 + 生成高性能 kernel。有三个工程细节值得单独记一下。

### Host Codegen

加速器越快，CPU 侧 launch overhead 越致命。V4 把大部分 host-side 逻辑（shape / dtype 校验、layout 检查）**从 Python 搬到生成的 host code 里**，基于 TVM-FFI 框架 + 零拷贝 tensor interop。

结果：**per-invocation 的 host validation overhead 从几十-几百 μs 降到 < 1 μs**。

### Z3 SMT Solver

TileLang compiler pass 里的整数表达式（layout inference、memory hazard detection、bound analysis）翻译成 Z3 的 quantifier-free non-linear integer arithmetic (QF_NIA)，用 SMT solver 证明优化需要的 invariant。

对 vectorization、barrier insertion、code simplification 多个 pass 都有影响。编译时间仅多几秒。

### Bitwise Reproducibility

数值严格度上，TileLang 默认禁用 fast-math；`T.__exp / T.__log / T.__sin` 是 opt-in 的近似算子；提供 `T.ieee_fsqrt / T.ieee_fdiv / T.ieee_add` 等 IEEE-754 严格算子；`T.annotate_layout` 让开发者钉死 layout，确保和 CUDA baseline bit-identical。

## Batch-Invariant & Deterministic Kernels

V4 有一个非常硬核的工程目标：**train / post-train / inference 三个阶段 bitwise 对齐**。意义：

- 调试稳定性问题时能精准定位数值源头
- Post-training 行为不受 batch 组合影响
- RL rollout 和 training 之间没有数值漂移

### Batch Invariance

同一个 token 在不同 batch 里输出要完全 bit 相同。实现里有两处关键挑战：

**Attention**。不能用 split-KV（FlashDecoding 的并行策略），它会引入不同的 reduction 顺序。V4 用 dual-kernel decoding：

| Kernel | 作用 |
|---|---|
| Kernel 1 | 整个 sequence 在单个 SM 完成，适合 wave 充满的情形 |
| Kernel 2 | Wave 最后没填满时，多个 SM 协作，accumulation 顺序匹配 Kernel 1，用 thread-block cluster 的 distributed shared memory 做高速数据交换 |

两个 kernel bitwise 一致，batch-invariant decoding 的 overhead 可以忽略。

**Matrix Multiplication**。cuBLAS 不满足 batch invariance，split-K 也不行（不同 split 数产生不同 reduction 顺序）→ **V4 全面用 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 替代 cuBLAS**。small batch 场景放弃 split-K，换一套优化做到匹配或超过 split-K 性能。

### Deterministic Backward

训练 backward 的非确定性主要来自 `atomicAdd` 的非结合性。V4 的对策：

| 部位 | 问题 | 解法 |
|---|---|---|
| Attention Backward | sparse attention 用 atomicAdd 累加 KV token 梯度 | 每个 SM 独立 accumulation buffer，最后做确定性全局 reduce |
| MoE Backward | 多 SM 从不同 rank 写到同一 receiving rank buffer 竞争 | per-rank token order preprocessing + buffer isolation |
| mHC Matmul | 输出维度 = 24（太小），必须用 split-K | 每份 split 单独输出，subsequent kernel 做 deterministic reduction |

## FP4 Quantization-Aware Training

### 哪里用 FP4

V4 的 FP4 不是推理量化，是 training-aware 的真 FP4。用在两个地方：

- **MoE expert weights**——GPU memory 的大头
- **CSA Indexer 的 QK 路径**——长上下文场景下 attention 打分这一步完全跑 FP4（activations、weights、matmul 全部 FP4）

Index scores `I_{t,s}` 本身从 FP32 降到 BF16：top-k selector 2× 加速，保持 **99.7% 的 KV entry recall**。

### MXFP4 和 lossless dequant

用的是 **MXFP4**（Rouhani et al. 2023，开放标准），不是 NVFP4。SGLang PR 里的 quantizer 文件命名 `mxfp4_deepseek.py` 印证了这一点。

一个容易被忽视的关键 trick：**FP4 → FP8 dequant 是无损的**。原因是：

- FP8（E4M3）比 FP4（E2M1）多 2 个 exponent bits
- 只要 FP4 sub-block（1 × 32 tiles）内的 scale factor 的动态范围不超过 FP8 的 dynamic range，fine-grained scale 信息可以完全被 FP8 吸收
- 经验证当前训练权重满足这个条件

后果：V4 整个 QAT pipeline **复用现有 FP8 mixed-precision framework 零修改**。Backward 对 FP8 weights 直接求导，梯度传回 FP32 master weights，等价于 STE (Straight-Through Estimator)。

### 对 Hopper / Blackwell 代差的含义

这个 trick 大幅缩小了 Hopper 和 Blackwell 的代差：

**Blackwell**：原生吃 FP4，推理时最省。

**Hopper（H100 / H200）**：dequant 到 FP8 是**无损的**，存储还是 FP4 的空间（memory traffic 省一半），算力走 FP8 Tensor Core。

架构篇里我之前写的"Hopper 代差拉大"要修正——**memory-bound 场景下 Hopper 吃到 V4 的大部分收益**，只在 FP4 原生 compute 密集场景才真拉开。

## 训练框架

训练框架继承自 V3 的 stack，但要适配 V4 的几个新组件：Muon、mHC、Hybrid Attention。有四个子模块值得展开。

### Muon 的 ZeRO 异构策略

Muon 需要**完整梯度矩阵**做 Newton-Schulz 正交化，和 ZeRO 的 partition parameter matrices 思路冲突。V4 的做法是 Hybrid ZeRO。

**Dense parameters**：

- 限制 ZeRO parallelism 最大尺寸
- 背包算法分配 parameter matrices 到 ranks，保证各 rank 负载均衡
- Bucket padding 匹配最大 bucket（memory overhead < 10%）
- 如果 data parallelism 超限，**在额外 DP group 上冗余计算 Muon update**（用算力换 bucket 内存）

**MoE parameters**：

- 每个 expert 独立优化
- Flatten 所有层所有 expert 的 down projections（SwiGLU 里），再 flatten up projection 和 gate matrices
- 对 flattened vector padding 到能均分到所有 rank（不切开单个 logical matrix）
- expert 数量大，不限 MoE 的 ZeRO parallelism，padding 开销可忽略

**其它优化**：

- 同 shape 连续参数自动 merge → Newton-Schulz batched 跑以提高 GPU 利用
- Newton-Schulz 用 BF16 matmul 仍稳定——利用这一点，MoE 梯度以 stochastic rounding 降到 BF16 做 DP all-reduce，通信量减半
- Two-phase reduce-scatter：先 all-to-all local gradients，再各 rank FP32 local sum，避免低精度 adder 累加误差

### mHC 实现优化

mHC 的 `n_hc × d` 扩展 residual stream 增加 activation memory 和 pipeline 通信。V4 的优化：

- Fused kernels（train + inference）
- Selective recomputation：重算大部分 hidden state 和 normalized layer input，避免重算 compute-heavy 操作
- 调整 DualPipe 1F1B overlap 适配增大的 pipeline comm

最终 mHC 的 wall-time overhead 压到 **6.7% of 1F1B pipeline stage**。

### Contextual Parallelism for 1M

常规 Context Parallelism 按 sequence 维度分 rank，每个 rank 管 `s` 个连续 token。V4 的 CP 碰到两个新问题：

- 压缩比 `m` 或 `m'` 不一定整除 `s`：各 rank 压缩长度不一致
- 压缩需要连续 `m` 个 KV entries，可能**跨 CP rank 边界**

V4 的两阶段通信：

1. 每个 `rank_i` 把本地最后 `m` 个 uncompressed KV entries 发给 `rank_{i+1}`
2. `rank_{i+1}` 把这些接收的 entries 和自己的 `s` 个本地 entries 合并压缩，产生固定长度 `s/m + 1` 的 compressed entries（带一些 padding entries）
3. All-gather 所有 rank 压缩后的 KV entries
4. Fused select-and-pad 算子重组成完整 compressed KV（总长 `cp_size × s/m`，padding 放尾部）

对 HCA 和 CSA 的 indexer，每个 query 的 compressed KV 可见范围可以 rule-based precompute；CSA 的 sparse attention 由 top-k selector 显式指定。

### Extended Autograd

常规 activation checkpointing 粒度太粗（整个 module），要么 recompute 太多、要么 memory 不够省。手写 forward + backward 能精细控制但代码复杂度爆炸。V4 走中间路线：**tensor 级别的 checkpointing，且保留 autograd**。

- 基于 TorchFX 追踪完整 computation graph
- 每个被 annotate 的 tensor → backward traversal 找到最小重算子图
- Recomputation 实现为**直接释放 annotated tensor 的 GPU memory + 复用 storage pointer**，不做 GPU memory copy
- Graph tracing 能自动识别 shared storage 的 tensor，做去重（比如 reshape 的输入输出共享底层 storage，只算一次）

开发者只需要 annotate 哪些 tensor 要 checkpoint，不用关心底层 memory 管理。

## 推理框架

### Heterogeneous KV Cache

V4 的 Hybrid Attention 让 KV cache **每层大小不同**——CSA 压 `m` 倍、HCA 压 `m'` 倍、Sliding Window Attention (SWA) 又保留 `n_win` 个未压缩 token。这完全打破了 PagedAttention 的一刀切 page 假设。

```
State Cache                  经典 KV Cache
┌─────────────────────┐     ┌──────────────────────┐
│ Request 1           │     │ Block 0              │
│ ┌─SWA KV─┐ ┌─未压缩─┐│    │ ┌Layer-2 CSA Indexer│
│ │        │ │ tail  ││    │ │Layer-2 CSA Main KV│
│ └────────┘ └───────┘│    │ └Layer-3 HCA KV──  │
│ Request 2           │     │ Layer-4 CSA...      │
│ ...                 │     │ Block 1, 2, ..., N  │
└─────────────────────┘     └──────────────────────┘
每 request 固定大小      每 block 容纳 lcm(m, m') 个原 token
per-position state     产生 k₁ = lcm(m,m')/m 个 CSA entries
                       产生 k₂ = lcm(m,m')/m' 个 HCA entries
```

图 3-1 V4 的异构 KV Cache 布局（对应 Tech Report Figure 6）
{: .figcaption }

分两类：

**State Cache**：存 SWA 的 KV + CSA/HCA 压缩分支里"尾巴上还没满 `m` 个 token"的 hidden state。这类完全由 request 当前位置决定，被当成 sequence state 管理，每 request 预分配固定大小。

**经典 KV Cache**：存 CSA 压缩 KV + HCA 压缩 KV + CSA Indexer 的 compressed keys。block 粒度 = `lcm(m, m')`，每 block 覆盖这么多原 token。

Sparse attention kernel 要和 KV layout 协同设计才能避免性能损失。传统高性能 attention kernel 假设每 block 固定 `B` 个 token，V4 需要 variable-tokens-per-block（任意 `lcm(m, m')` 的倍数）。kernel 侧的 cache-line padding 需要和 cache layout 对齐。

**对 vLLM / SGLang 的冲击**：PagedAttention 那套"统一 page"的假设不适用。SGLang day-0 PR 里用的是 `deepseek_v4_backend_radix.py`——RadixAttention 变种，**绕开了 PagedAttention 的直接路径**。

### On-Disk KV Cache

V4 的部署假设**磁盘 backed KV cache 是常态，不是 fallback**。对 CSA/HCA compressed KV 和 SWA uncompressed KV 各自设计了磁盘存储策略。

核心动机是 shared-prefix reuse——长 ctx + 多租户场景下，prefix 重用能避免大量重复 prefill。具体实现指向 **3FS**（DeepSeek 自家的分布式文件系统，[github.com/deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)）。

## RL 与 OPD 基础设施

Post-training 从 V3.2 的 "一锅炖 RL" 改成 **Specialist → Unified via On-Policy Distillation**。配套 infra 有五件硬核工程。

### FP4 Integration

Rollout 和 teacher inference 直接用真 FP4 权重（和线上部署一致）；training 模拟 FP4 via 前面说的 lossless FP4 → FP8 dequant。

### Full-Vocabulary OPD Teacher Scheduling

挑战是：10+ teacher 模型，每个可能 trillion-scale 参数；vocab size > 100k；所有 teacher 同时在线不可行。V4 的解法分几层：

| 优化 | 细节 |
|---|---|
| Teacher 权重 offload | 全部存到中心化分布式存储，on-demand 加载；用 ZeRO-like sharding 缓解 I/O 和 DRAM 压力 |
| Logit 不 materialize | 只缓存 **last-layer teacher hidden states** 到中心 buffer；训练时再过对应的 prediction head 重建 full logits |
| Sample 按 teacher 排序 | 每个 mini-batch 只装一个 teacher head，per-head per-mini-batch 只加载一次 |
| 异步加载 | 所有 parameter / hidden-state load/offload 后台异步，不阻 critical path |
| TileLang KL kernel | Teacher / student logit 的 KL 散度用专门的 TileLang fused kernel 算，省内存 + 加速 |

### Preemptible & Fault-Tolerant Rollout

大集群硬件故障常态化，RL rollout 还要支持高优先级任务打断。V4 的解法是**token-granular Write-Ahead Log (WAL)**：

- 每 token 生成立即 append 到 request 的 WAL
- Preemption：暂停 engine + 保存 KV cache
- Resume：用 WAL + KV cache 继续 decode
- 硬件 fatal error：用 WAL 的 tokens 重跑 prefill 重建 KV cache

一个容易被忽视的陷阱：**不能从头 regenerate 未完成 request**。因为短响应更容易"活过"中断，regenerate from scratch 会让 model 产生**更短的 sequence 偏差**（length bias）。这是 inference-as-service 做 RL rollout 的一个微妙问题。

### 1M ctx 的 RL 数据 loader

1M 序列数据的内存压力极大，V4 把 rollout 数据拆成：

- **轻量 metadata**：全量加载到 dispatch 端做 global shuffling 和 packing 计算
- **Heavy per-token fields**：用 shared-memory data loader 消除 intra-node 数据冗余；mini-batch 粒度消费完立即释放

每 device mini-batch 数量根据负载动态调整，在 compute throughput 和 I/O overlap 之间找平衡。

### DSec Sandbox

V4 agentic post-training 和评测用的 sandbox platform 叫 **DeepSeek Elastic Compute (DSec)**。Rust 三件套（`Apiserver` gateway + `Edge` per-host agent + `Watcher` cluster monitor），横向扩展在 3FS 上，单集群管**数十万并发 sandbox**。

同一个 Python SDK `libdsec` 下暴露了四种执行底座：

| 底座 | 实现 | 特点 |
|---|---|---|
| Function Call | pre-warmed container pool | 最轻，无 cold-start，无状态 invocation |
| Container | Docker + EROFS 3FS layered storage | 镜像按需加载，3FS 做只读 base layer |
| microVM | Firecracker | VM 级隔离，高密度 |
| fullVM | QEMU + overlaybd | 支持任意 OS |

三个工程挑战值得单独提一下：

**快速镜像加载**。基于 3FS 的 EROFS overlay layer。镜像 metadata 预热到本地磁盘，数据块按需从 3FS 拉取。microVM 用 overlaybd——只读 base 在 3FS 跨 instance 共享，写入走本地 copy-on-write。毫秒级 resume。

**高并发密度**。去重虚拟化环境的 page cache 减少冗余内存；反向 memory reclamation 允许 safe overcommit；减少 container runtime 的 spinlock 竞争。

**Trajectory Logging**。每个 sandbox 全局有序的 trajectory log 有三个用途：preempt 后 resume 时 replay cached results（避免重复执行 non-idempotent 操作）；细粒度 provenance 追溯；deterministic replay。

对做 agentic RL 基础设施的队伍，DSec 是一个相对完整的开源参考——特别是 **3FS + EROFS 分层存储 + preemption-safe 的 trajectory replay** 这条路。

## Post-Training 方法学

Infra 之外，Section 5.1 的方法学上也有几处不寻常的设计。

### Specialist + On-Policy Distillation

两阶段：

```
Base Model ─┬─► SFT + RL(GRPO) on Math    ──► Math Specialist ─┐
            ├─► SFT + RL(GRPO) on Code    ──► Code Specialist  ─┤
            ├─► SFT + RL(GRPO) on Agent   ──► Agent Specialist ─┼─► OPD ─► V4
            ├─► SFT + RL(GRPO) on IF      ──► IF Specialist    ─┤
            └─► ... (10+ domains)                               ─┘
```

图 3-2 V4 post-training 两阶段：先 per-domain specialist，再 multi-teacher OPD 蒸馏到单一 unified 模型
{: .figcaption }

OPD objective：

$$\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$$

其中 `π_θ` 是 student（unified）policy，`π_{E_i}` 是第 `i` 个 expert。**On-policy**——trajectory 从 student 采样，保证 student 只从 context-relevant expert 学（math task 只学 math expert）。V4 用的是 **full-vocabulary logit distillation**（不是 token-level KL 估计），variance 更低、蒸馏更稳。

### Generative Reward Model

对 hard-to-verify task（没有规则 verifier），V4 不走 RLHF，改走 GRM：

- Actor 网络**同时扮演 generator 和 judge（GRM）**
- RL optimization 直接 apply 到 GRM 本身
- Reasoning capability 自然融入 evaluation process
- 只需少量人类标注（rubric-guided data），而不是大规模 preference dataset

### Tool-Call Schema

V4 抛弃了 JSON-based tool call，改用 XML-style 的 `<|DSML|>` special token：

```xml
<|DSML|tool_calls>
  <|DSML|invoke name="$TOOL_NAME">
    <|DSML|parameter name="$PARAM_NAME" string="true|false">$VALUE</|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>
```

动机是 JSON escaping 在复杂 tool call 里容易失败；XML + special tokens 减少 tool-call error。

### Interleaved Thinking

V3.2 每次收到新 user message 就丢弃 thinking trace；V4 在 tool-calling 场景下**保留全部 reasoning 跨 turn**（利用 1M ctx），让 agent 能保持 coherent 累积 chain of thought。普通对话仍然丢弃以节省 token。

### Quick Instruction

常见 chatbot 需要先做一些 auxiliary task（决定是否 web search、识别意图、生成搜索 query）。V3.2 用独立的小模型做，每次要额外 prefill。V4 引入 Quick Instruction：

| Special Token | 作用 |
|---|---|
| `<\|action\|>` | 判断是否需要 web search |
| `<\|title\|>` | 生成对话 title |
| `<\|query\|>` | 生成搜索 query |
| `<\|authority\|>` | 分类 prompt 对权威性的需求 |
| `<\|domain\|>` | 识别 prompt 领域 |
| `<\|extracted_url\|>` / `<\|read_url\|>` | 判断 URL 是否要抓取 |

这些 token 直接附加到输入序列尾部，**复用已经计算的 KV cache**，避免独立小模型的重复 prefill，TTFT 显著下降。

## 结论与官方自评

Section 6 的诚实度值得专门记一下。

### 官方承认"架构偏复杂"

> "To minimize risk, we retained many preliminarily validated components and tricks, which, while effective, made the architecture relatively complex. In future iterations, we will carry out more comprehensive and principled investigations to distill the architecture down to its most essential designs, making it more elegant without sacrificing performance."

翻译下就是：V4 里很多设计（CSA 的 overlapping compression、Lightning Indexer、mHC 的 Sinkhorn-Knopp、Hybrid Newton-Schulz 系数）都是"保险起见留着的"。后续会精简。

### 承认几个机制原理未明

> "Anticipatory Routing and SwiGLU Clamping have been proven effective in mitigating training instabilities, their underlying principles remain insufficiently understood."

Anticipatory Routing 和 SwiGLU Clamping 在训练稳定性上有效但原理不清。而且 Section 2 里完全没提这两个东西——**说明还有 trick 没公开**。

### 官方 roadmap

1. 更稀疏的 embedding 模块（Cheng et al. 2026）——暗示 MoE / 稀疏 attention 之外继续稀疏化
2. 低延迟架构——1M ctx 的部署响应要能做交互式
3. 长 horizon multi-round agentic tasks（对应 DSec 的方向）
4. 多模态（暂时是 text-only）
5. 更好的 data curation

## 对 NV infra 的几个判断

这一节是我自己的 take，不是 Tech Report 内容。

**马上能用上的**。MegaMoE kernel（DeepGEMM PR #304）对自家 EP 部署是 drop-in 收益。FP4 → FP8 lossless dequant 这条让 Hopper (H100/H200) 不必被 FP4 代差拍死。Heterogeneous KV cache 思路对 vLLM / SGLang / TRT-LLM 的 PagedAttention 一刀切假设是直接挑战——SGLang PR #23600 用 RadixAttention 变种是务实选择。

**给 CUDA kernel 生态的信号**。TileLang 可能挤占 CUTLASS + cuBLAS 在 MoE / attention 场景的生态位。TileLang 不是 NV 主推，但 DeepSeek 的规模化验证给了它很强背书。NV 的应对要么是 Triton 加固、要么是官方出更强的 tile-level DSL。DeepGEMM 替代 cuBLAS 的趋势也值得 NVIDIA 的 cuBLAS 组关注——small batch、batch-invariance、split-K 限制几个场景 cuBLAS 的覆盖度不够。

**给硬件设计的直接反馈**。Tech Report 3.1 末段等于 DeepSeek 给 NV / AMD / 华为的一份甲方需求书。核心结论：`C/B ≤ 2d = 6144 FLOPs/Byte`——到了这个比例之后不要再堆带宽。Power headroom、low-latency cross-GPU signaling、hardware-friendly activation（替代 SwiGLU）是同等重要的三个补充。

**EP / PD 分离叙事的延续**。V4 把 `C/B` 目标推到 6144 → 未来 EP 部署不再"越大带宽越好"，要算清楚 comp/comm ratio。1M ctx KV cache 10% + On-Disk 存储 → PD 分离 + KV mover 的带宽压力进一步降。NIXL / Dynamo KV Router / DeepEP v2 + NCCL GIN 这条链正好对接。

**Agentic infra 的参考**。DSec 对做 agentic RL 基础设施的队伍是硬参考实现。3FS + EROFS + microVM 这条栈可以直接借鉴。NV 自己的 Nemo-Agent 这类项目可以看看能否整合。

## 还没覆盖

Tech Report Section 4（Pre-Training：data construction / training setup / mitigating instability）架构篇和这篇都还没读。Anticipatory Routing 和 SwiGLU Clamping 的具体 recipe 在 Section 4.2.3 提到一点但不足以复现，需要等续作或社区复现。

Section 5.3（Standard Benchmark Evaluation）和 Section 5.4（Performance on Real-World Tasks）的完整数据（MRCR 长上下文曲线、Terminal Bench 细节、Chinese writing / white-collar 的 human eval）架构篇里只列了核心 benchmark table，完整 per-mode 的对比还可以再展开。

后续有进展再更新。

## 相关链接

一手：

- [Tech Report PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)（Section 3 / 5 / 6）
- [MegaMoE in DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/pull/304)
- [3FS](https://github.com/deepseek-ai/3FS)
- [TileLang](https://github.com/tile-ai/tilelang)（Wang et al., ICLR 2026）
- [mHC paper](https://arxiv.org/abs/2512.24880)（Xie et al., 2026）

相关：

- [Muon is scalable for LLM training](https://arxiv.org/abs/2502.16982)（Liu et al., Kimi team 2025）
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation)（Lu & TM Lab, 2025）
- [Hash Layers for Large Sparse Models](https://proceedings.neurips.cc/paper/2021/hash/92bf5e6240737e0326ea59846a83e076-Abstract.html)（Roller et al., NeurIPS 2021）

生态：

- [SGLang day-0 PR #23600](https://github.com/sgl-project/sglang/pull/23600)

KB 内：

- 架构篇：[DeepSeek-V4：MLA 彻底不用了？](2026-04-24-deepseek-v4.md)
- [DeepEP 项目笔记](../../projects/deepep.md)
- [NCCL GIN](2026-04-23-nccl-gin.md)
- [AFD Challenges](2026-04-20-afd-challenges.md)
