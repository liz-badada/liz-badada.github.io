---
title: Communication & Networking
tags: [communication, nccl, networking]
---

# Communication & Networking

The glue between parallelism and hardware. Often what separates 40% MFU from 60%.

## Sub-areas

### Collective operations
- **all-reduce** — DP gradient sync, TP forward.
- **reduce-scatter + all-gather** — ZeRO / FSDP.
- **all-to-all** — MoE dispatch, EP, sequence parallel redistribution.
- **point-to-point send/recv** — PP micro-batches.

### Libraries & algorithms
- **NCCL** — tree / ring / double-binary-tree.
- **MSCCL / MSCCL++** — synthesized collectives, GPU-initiated comm.
- **DeepEP** — DeepSeek's EP-optimized all-to-all.
- **NIXL** — NVIDIA open-source comm library for inference (KV transfer).

### Interconnect
- **NVLink / NVSwitch** — intra-node, ~900 GB/s on H100 / ~1.8 TB/s on B200.
- **InfiniBand** (HDR/NDR) — 200–400 Gbps, RDMA.
- **RoCE / Ethernet** — cheaper, harder to tune.
- **UEC** (Ultra Ethernet Consortium) — emerging standard.

### Compute-comm overlap
- Non-blocking collectives, CUDA streams, graph capture.
- **Centauri**, **T3**, **CoCoNet** — fused comp+comm kernels.
- **Interleaved 1F1B**, **DualPipe** — hide PP bubbles.
- SM-aware scheduling: reserve SMs for comm.

### Topology-aware placement
- Rail-optimized topologies for all-to-all.
- Placement within a DGX / NVL72 vs across pods.

## Key metrics

| Metric | Why |
|---|---|
| Bus BW utilization | Are you saturating NVLink / IB? |
| Comm % of step time | If > 30%, rethink parallelism degree |
| Tail latency of collectives | Straggler signal |

## Canonical references

- *NCCL* design docs
- *MSCCL* (Cowan et al., ASPLOS 2023)
- *DeepEP* (DeepSeek, 2025)
- *Centauri / T3 / CoCoNet*
