---
title: MoE Systems
tags: [moe, expert-parallelism]
---

# MoE Systems

Mixture-of-Experts. Sparse activation is easy on paper, painful on silicon.

## Sub-areas

### Routing
- **Top-k token choice** (classic).
- **Expert choice** (Zhou et al.) — experts pick tokens, balances load.
- **Shared + routed experts** (DeepSeek-V2/V3, Qwen1.5-MoE).
- **Fine-grained experts** — many tiny experts vs few big ones.

### Load balancing
- Auxiliary loss (importance / load).
- **DeepSeek's auxiliary-loss-free** bias update scheme.
- Dropping vs padding vs capacity factor.
- Skew under real traffic (some experts hot all day).

### Expert parallelism (EP)
- All-to-all dispatch / combine — usually the bottleneck.
- **DeepEP** — FP8 / NVLink-aware all-to-all. v2 rewrite unifies HT/LL into `ElasticBuffer`, switches RDMA backend to NCCL GIN, scales to EP 2048. See [project note → DeepEP](../projects/deepep.md).
- EP + TP + DP mixing.

### MoE inference
- Prefill vs decode behavior (batch size × expert usage).
- KV cache is still dense — MoE helps FFN, not attention.
- Expert-level prefetch / placement.

### MoE training
- Checkpoint size explosion.
- Expert-parallel gradient sync.
- Upcycling from dense (Qwen1.5-MoE approach).

## Recent notes

- **DeepSeek-V4** (2026-04-24) — MLA dropped; Hybrid Attention = CSA (sequence-dim compression + DSA) + HCA (heavier compression, dense); Shared-KV MQA; mHC residuals; Muon optimizer; FP4+FP8 mixed training; 1M native context; KV cache ≈10% of V3.2. See [DeepSeek-V4：MLA 彻底不用了？](../papers/posts/2026-04-24-deepseek-v4.md).

## Canonical references

- *GShard* (Lepikhin et al., 2020)
- *Switch Transformer* (Fedus et al., 2021)
- *Expert Choice Routing* (Zhou et al., 2022)
- *DeepSeek-V2 / V3* — auxiliary-loss-free MoE
- *DeepSeek-V4* — Hybrid CSA+HCA attention, mHC, Muon, FP4 training
- *DeepEP* — comm kernels
- *Mixtral of Experts* (Jiang et al., 2024)
