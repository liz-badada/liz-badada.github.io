---
title: Perf Modeling & Simulation
tags: [perf-modeling, simulation]
---

# Perf Modeling & Simulation

Predict before you build. A good analytical model tells you which parallelism is viable on which hardware before burning a single GPU-hour.

## Sub-areas

### First-principles analysis
- **Roofline model** — arithmetic intensity vs bandwidth.
- Compute-bound vs memory-bound per kernel.
- Communication roofline (interconnect BW × overlap).

### Analytical simulators
- **AIConfigurator** (NVIDIA) — search parallelism configs for TRT-LLM serving.
- **Calculon**, **Astra-sim**, **LLMCompass** — training perf prediction.
- **Frontier-like simulators** — cluster-level workload forecasting.

### Cycle / instruction level
- **GPGPU-Sim**, **Accel-Sim**.
- Hopper-aware simulation still an open problem.

### What good perf models answer

1. Best (TP, PP, DP, EP) mix for this model on this cluster.
2. Which batch size maximizes goodput@SLO.
3. Whether FP8 is worth the engineering cost here.
4. Break-even point for P/D disaggregation.
5. Where comm-compute overlap hurts.

### Things perf models get wrong

- Kernel fusion effects.
- Cache hierarchy behavior for attention.
- Straggler / failure impact at scale.
- Contention on shared interconnect.

## Your existing notes

- [AIConfigurator framework deep-dive](../projects/aiconfigurator.md)
- [Frontier simulator paper notes](../papers/posts/2026-04-09-frontier-simulator.md)

## Canonical references

- *Roofline: An Insightful Visual Performance Model* (Williams et al., 2009)
- *Calculon* (Isaev et al., SC 2023)
- *LLMCompass* (Zhang et al., HPCA 2024)
- *Astra-sim 2.0* (Won et al., ISPASS 2023)
