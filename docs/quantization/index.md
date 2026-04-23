---
title: Quantization & Compression
tags: [quantization, compression]
---

# Quantization & Compression

Shrink weights, activations, and KV. Every 2× compression unlocks 2× batch or 2× concurrency.

## Sub-areas

### Weight-only quantization
- **GPTQ** — second-order, post-training.
- **AWQ** — activation-aware, search-based.
- **SpQR / QuIP / SpinQuant / OmniQuant** — rotation / smoothing tricks.

### Weight + activation quantization
- **SmoothQuant** — redistribute outliers to weights.
- **LLM.int8()** — mixed-precision outliers.
- **FP8** (E4M3 / E5M2) — native on Hopper / Blackwell.
- **FP4** (MXFP4, NVFP4) — Blackwell.

### KV cache quantization
- FP8 / INT8 / INT4 KV.
- **KIVI**, **KVQuant**.

### Sparsity & pruning
- Structured sparsity (2:4 on Ampere+).
- Activation sparsity (**Deja Vu**, contextual sparsity).
- Weight pruning (**Wanda**, **SparseGPT**).

### Low-bit training
- FP8 training (Transformer Engine).
- MXFP8 mixed precision.

## Trade-off axes

| Axis | Tension |
|---|---|
| Accuracy drop | LLM benchmarks often understate real drop |
| Kernel availability | FP4 great in theory, kernel support lagging |
| Calibration cost | AWQ / GPTQ need data and time |

## Canonical references

- *GPTQ* (Frantar et al., 2022)
- *AWQ* (Lin et al., 2023)
- *SmoothQuant* (Xiao et al., 2022)
- *FP8 Formats for Deep Learning* (NVIDIA)
