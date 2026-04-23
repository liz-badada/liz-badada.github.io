---
title: Parallelism
tags: [parallelism, distributed]
---

# Parallelism

How to split a model across devices. Pick the right mix of parallelism axes for your arithmetic intensity, memory footprint, and network topology.

## The basic axes

| Axis | Splits | Comm pattern |
|---|---|---|
| **DP** Data | batch | all-reduce on grad |
| **TP** Tensor | hidden dim | all-reduce per layer (Megatron-TP) |
| **PP** Pipeline | layers | point-to-point, needs micro-batching |
| **SP** Sequence | seq dim of norm/dropout | paired with TP |
| **CP** Context | attention seq dim | Ring-Attention / stripe |
| **EP** Expert | MoE experts | all-to-all |
| **ZeRO / FSDP** | optimizer/grad/param shards | all-gather / reduce-scatter |

## Advanced patterns

- **3D / 4D parallelism** — combinations (e.g. TP×PP×DP×EP).
- **Hybrid sharding** (FSDP shard inside group, replicate across groups).
- **AFD (Attention-FFN Disaggregation)** — run attention and FFN on different device pools to decouple arithmetic intensity and memory pressure. [Your current research focus.]
- **Zero-bubble pipelines** — interleaved 1F1B, DualPipe.

## How to pick

1. Does the model fit on one GPU? No → need sharding (TP / FSDP / PP).
2. Is compute or comm the bottleneck? Roofline first.
3. Is the network NVLink-only or IB-crossing? TP needs fast intra-node, DP is fine on IB.
4. MoE? Add EP, watch for skew.
5. Long context? Add CP.

## Canonical references

- *Megatron-LM* (Shoeybi et al., 2019) — TP origin
- *GPipe*, *PipeDream*, *Megatron 1F1B* — PP
- *ZeRO* (Rajbhandari et al., SC 2020), *FSDP*
- *Ring Attention* (Liu et al., 2023) — CP
- *DeepSpeed-MoE*, *GShard* — EP
