---
title: Long Context
tags: [long-context, attention]
---

# Long Context

Getting past 128k tokens cheaply. The math is quadratic, the memory is linear-per-token, both bite.

## Sub-areas

### Training-side
- **Ring Attention** (Liu et al.) — CP with overlapped KV shuffling.
- **Context Parallelism** (NVIDIA, Megatron) — similar idea, productionized.
- **Sequence Parallelism** — for norm / dropout only.
- **LongRoPE / YaRN / NTK** — position encoding extrapolation.

### Inference-side
- **Paged KV** at 1M tokens — HBM alone isn't enough → offload / disagg.
- **StreamingLLM** — attention sink + sliding window.
- **H2O / Scissorhands / SnapKV** — token eviction.
- **Infini-attention**, **Landmark Attention** — compressed history.
- **DuoAttention** — retrieval heads vs streaming heads.

### Sparse / linear / hybrid
- **Mamba / Mamba-2** — SSM.
- **RWKV**.
- **Jamba** — hybrid Mamba + attention.
- **Native Sparse Attention** (DeepSeek, 2025).

### Retrieval-augmented
- RAG as context compression.
- KV cache reuse across RAG queries.

## Eval traps

- "1M context" claims often measured only on needle-in-haystack.
- RULER, LongBench, InfiniteBench — harder evals.
- Real workloads (codebases, books) stress differently.

## Canonical references

- *Ring Attention* (Liu et al., 2023)
- *StreamingLLM* (Xiao et al., 2023)
- *H2O* (Zhang et al., 2023)
- *Infini-attention* (Munkhdalai et al., 2024)
- *Native Sparse Attention* (DeepSeek, 2025)
