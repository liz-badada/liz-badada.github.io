---
title: Hardware & Systems
tags: [hardware, gpu, systems]
---

# Hardware & Systems

The silicon and interconnect that everything else runs on.

## NVIDIA GPUs

| GPU | Arch | FP8 TFLOPS | HBM | NVLink BW |
|---|---|---|---|---|
| A100 | Ampere | — (FP16 312) | 80 GB HBM2e | 600 GB/s |
| H100 | Hopper | 2000 (sparse 4000) | 80 GB HBM3 | 900 GB/s |
| H200 | Hopper | 2000 | 141 GB HBM3e | 900 GB/s |
| B200 | Blackwell | 4500 FP8 (9000 FP4) | 192 GB HBM3e | 1800 GB/s |
| GB200 NVL72 | Blackwell | 72 B200 in one NVLink domain | — | — |

### Hopper features
- Thread Block Clusters, Distributed Shared Memory.
- Tensor Memory Accelerator (TMA).
- WGMMA async tensor ops.
- FP8 (E4M3 / E5M2).

### Blackwell features
- Second-gen Transformer Engine, FP4 (MXFP4, NVFP4).
- Larger NVLink domain (NVL72 = 72 GPUs).
- Decompression engine.

## Interconnect

- **NVLink / NVSwitch** — intra-node / intra-rack.
- **InfiniBand** HDR / NDR / XDR — cross-node.
- **RoCE** / UEC — Ethernet alternatives.
- **SHARP** — in-network reduction.

## CUDA stack

- **CUDA 12.x / 13.x** — driver + runtime.
- **cuBLAS, cuDNN** — dense ops.
- **NCCL** — collectives.
- **CUTLASS** — template kernels.
- **Triton** — Python kernel DSL.
- **TensorRT / TensorRT-LLM** — compiled inference.

## Non-NVIDIA

- **AMD MI300X / MI325X** — CDNA3, RCCL, ROCm.
- **Google TPU v5p / Trillium** — XLA / JAX.
- **AWS Trainium / Inferentia** — Neuron SDK.
- **Cerebras WSE**, **Groq LPU**, **SambaNova** — exotic.

## Canonical references

- NVIDIA GPU architecture whitepapers (Hopper, Blackwell)
- Horace He's *Making Deep Learning Go Brrrr From First Principles*
- MLPerf results for cross-vendor perf reality
