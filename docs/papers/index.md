---
hide:
  - navigation
---

# Papers

Dated reading notes. Each note carries frontmatter: `arxiv`, `venue`, `tags`, `categories`, `tier (L1–L5)`, `status`.

## Recent notes

| Date | Title | Category | Tags | Tier |
|---|---|---|---|---|
| 2026-04-24 | [DeepSeek-V4 基础设施拆解：6144 FLOPs/Byte 是怎么来的？](posts/2026-04-24-deepseek-v4-infra.md) | `large-scale` | deepseek · infra · ep-overlap · tilelang · fp4-qat · rl-infra · sandbox | L3 |
| 2026-04-24 | [DeepSeek-V4：MLA 彻底不用了？](posts/2026-04-24-deepseek-v4.md) | `moe` | deepseek · moe · hybrid-attention · fp4 · muon · mhc | L3 |
| 2026-04-23 | [NCCL GIN：把 device-initiated RDMA 带回 NCCL 生态](posts/2026-04-23-nccl-gin.md) | `communication` | nccl · gin · rdma · moe · deepep | L3 |
| 2026-04-20 | [AFD 真的通用吗？标准集群 + 细粒度 MoE 下的 dead zone](posts/2026-04-20-afd-challenges.md) | `parallelism` | afd · moe · roofline | L4 |
| 2026-04-09 | [AFD 的 r* 最优配比有闭式解吗？](posts/2026-04-09-afd-optimal-ratio.md) | `parallelism` | afd · perf-modeling | L4 |
| 2026-04-09 | [Frontier：为什么现有仿真器撑不起分离式推理？](posts/2026-04-09-frontier-simulator.md) | `perf-modeling` | simulation · moe · disagg | L3 |

> Full archive (paginated, auto-sorted by date) is in the **Archive** link in the right sidebar. **Categories** (also in the sidebar) filters by the 19 topic categories.

## The 19 categories

| Cluster | Categories |
|---|---|
| **Inference** | [engines](../engines/index.md) · [kv-cache](../kv-cache/index.md) · [attention](../attention/index.md) · [scheduling](../scheduling/index.md) · [quantization](../quantization/index.md) |
| **Parallelism & Comm** | [parallelism](../parallelism/index.md) · [large-scale](../large-scale/index.md) · [communication](../communication/index.md) · [moe](../moe/index.md) · [long-context](../long-context/index.md) |
| **Training & RL** | [training](../training/index.md) · [post-training](../post-training/index.md) · [agentic-rl](../agentic-rl/index.md) |
| **Applications** | [agents](../agents/index.md) · [multimodal](../multimodal/index.md) · [diffusion](../diffusion/index.md) |
| **Foundations** | [hardware](../hardware/index.md) · [perf-modeling](../perf-modeling/index.md) · [benchmarks](../benchmarks/index.md) |

## Reading tiers

| Tier | Meaning |
|---|---|
| **L1** | Aware it exists |
| **L2** | Abstract + intro + conclusion |
| **L3** | Full read + key experiments understood |
| **L4** | Can reproduce key figures / numbers |
| **L5** | Applied to own work |
