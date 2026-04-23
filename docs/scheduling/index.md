---
title: Scheduling & Batching
tags: [scheduling, batching, inference]
---

# Scheduling & Batching

Throughput-latency trade-off lives here. Bad scheduling wastes more FLOPs than bad kernels.

## Sub-areas

### Batching policies
- **Static batching** — legacy; padding kills throughput.
- **Continuous batching** — Orca-style iteration-level.
- **Chunked prefill** — blend prefill into decode batches to reduce TTFT variance.

### Speculative decoding
- **Draft model** (classic) — small model drafts, big verifies.
- **Medusa / EAGLE / EAGLE-2** — multi-head / tree drafting.
- **Lookahead decoding** — Jacobi iteration, no draft model.
- **REST** — retrieval-based drafting.

### Disaggregated P/D
- **DistServe**, **Splitwise**, **Mooncake** — prefill and decode pools with different hardware / batch sizes.
- KV transfer: RDMA over NVLink, PCIe, IB.

### SLO-aware scheduling
- Priority queues, fair sharing, preemption.
- TTFT vs TPOT trade-offs.
- Multi-LoRA routing.

## Key metrics

| Metric | Why |
|---|---|
| TTFT (P50/P99) | First-token latency for user experience |
| TPOT (P50/P99) | Per-token decode cost |
| Goodput@SLO | Throughput under latency constraint — the real metric |

## Canonical references

- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (Yu et al., OSDI 2022)
- *DistServe*, *Splitwise*, *Mooncake*
- *EAGLE-2* (Li et al., 2024)
- *SARATHI* / chunked prefill
