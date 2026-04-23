---
title: Attention & Kernels
tags: [attention, kernels, cuda]
---

# Attention & Kernels

The hot path. Anything that changes FLOPs/token or memory traffic on attention shows up here.

## Sub-areas

### Attention algorithms
- **FlashAttention v1/v2/v3** — tiled softmax, online rescaling, Hopper WGMMA / async copies.
- **FlashDecoding / FlashDecoding++** — split-KV during decode for low batch.
- **FlashInfer** — prefill + decode + paged + spec decode in one library.
- **FlashMLA** — Multi-head Latent Attention kernel (DeepSeek).
- **Paged / block-sparse attention** — sparse masks at block granularity.

### Attention variants
- **MHA / MQA / GQA** — head-count vs KV sharing trade-off.
- **MLA** (Multi-head Latent Attention, DeepSeek) — low-rank KV projection.
- **Linear / SSM / Mamba** — sub-quadratic alternatives.
- **Sliding window / StreamingLLM** — bounded-memory approximations.

### Kernel toolchains
- **CUTLASS** (3.x, CuTe) — template-heavy, Hopper-aware.
- **Triton** — Python-level kernel authoring.
- **ThunderKittens**, **CUTE-DSL** — emerging.

## Key metrics

| Metric | Why |
|---|---|
| TFLOPs / HBM BW utilization | How close are we to roofline |
| Kernel-only vs end-to-end latency | Framework overhead exposure |
| Numerical stability (BF16 / FP8) | Accuracy regressions |

## Canonical references

- *FlashAttention* (Dao et al., NeurIPS 2022)
- *FlashAttention-2* (Dao, 2023), *FlashAttention-3* (2024)
- *FlashInfer* (Ye et al., 2024)
- *DeepSeek-V2 / V3* — MLA
