---
title: Inference Engines
tags: [engines, inference]
---

# Inference Engines

The orchestrator layer: takes requests, schedules batches, owns KV, calls kernels, returns tokens.

## What to track per engine

| Axis | Questions to answer |
|---|---|
| Batching | Static / continuous / chunked prefill? Max batch? |
| KV management | Paged? Prefix cache? Multi-tier (GPU/CPU/SSD)? |
| Parallelism | TP / PP / EP / CP supported? Heterogeneous? |
| Quantization | FP8 weights / KV? INT4? SmoothQuant? |
| Scheduling | Priority, preemption, SLO-aware? |
| Spec decode | Medusa / EAGLE / draft-model / Lookahead? |
| Runtime | Custom / TensorRT / PyTorch-eager / CUDA graphs? |
| Multi-LoRA | Served concurrently? |

## Engines in scope

- **vLLM** — de-facto open source serving engine; PagedAttention origin.
- **SGLang** — RadixAttention prefix cache, strong for agent / structured workloads.
- **TensorRT-LLM** — NVIDIA first-party; best raw perf on NVIDIA GPUs.
- **LMDeploy** — Shanghai AI Lab; TurboMind backend, strong INT4.
- **TGI** — HuggingFace; simpler, Rust server.
- **llama.cpp / MLC / ExecuTorch** — edge / on-device.
- **Mooncake** — Moonshot's disaggregated KV design.

## Projects with full write-ups

See [Projects → Inference engines](../projects/index.md).

## Canonical references

- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., SOSP 2023) — vLLM origin paper.
- *SGLang: Efficient Execution of Structured Language Model Programs* (Zheng et al., 2024)
- *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (Qin et al., 2024)
