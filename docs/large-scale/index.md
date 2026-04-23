---
title: Ultra-scale Training
tags: [large-scale, distributed, fault-tolerance]
---

# Ultra-scale Training

10k+ GPU regime. Everything that's a rounding error at 100 GPUs becomes a daily problem here.

## What changes at scale

| Problem | Sub-1k GPU | 10k+ GPU |
|---|---|---|
| Failure rate | hours-days MTBF | many per day |
| Straggler variance | tolerable | dominant |
| Collective latency | fast | bubble growth |
| Checkpoint IO | minutes | tens of minutes |
| Power / thermal | uniform | regional brownouts |

## Sub-areas

### Fault tolerance & recovery
- **Async checkpointing** (DeepSpeed, Torch DCP).
- **In-memory checkpoint** (Gemini, CheckFreq).
- **Checkpoint resharding** — restart with different parallelism degree.
- **Elastic training** — resume with fewer nodes when some fail.

### Straggler / slow-node handling
- Detection, quarantine, redundant compute.
- Work stealing in PP.
- **Oobleck**, **Bamboo**, **Gemini**.

### Scheduler & cluster infra
- Job topology placement (**Themis**, **Shockwave**).
- Gang scheduling, checkpoint-aware preemption.
- Hot-swap replacement of failed nodes.

### Engineering observability
- Real-time loss / gradient norm monitors.
- NCCL hangs detection, ring health.
- Per-GPU power / thermal dashboards.

### Case studies
- **LLaMA-3 training paper** (Meta, 2024) — 16k H100s, failure log.
- **GPT-4 / Gemini / Claude** infrastructure write-ups (sparse but useful).
- **MegaScale** (ByteDance, NSDI 2024).

## Canonical references

- *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs* (Jiang et al., NSDI 2024)
- *The LLaMA 3 Herd of Models* (Meta AI, 2024) — training infra section
- *ReCycle*, *Oobleck*, *Bamboo*
