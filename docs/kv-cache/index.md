---
title: KV Cache & Memory
tags: [kv-cache, memory, inference]
---

# KV Cache & Memory

Where most of the cost and latency lives during decode. The center of gravity for modern inference systems.

## Sub-areas

### 1. Allocation & layout
- **PagedAttention** (vLLM) — block-based allocator, no contiguous requirement.
- **RadixAttention** (SGLang) — prefix tree for cache reuse across requests.
- **vAttention** — CUDA virtual memory for contiguous-looking paging.

### 2. Reuse (prefix caching)
- System prompt / few-shot / tool-spec sharing.
- Multi-turn chat reuse.
- Cross-request prefix hit rate as a scheduling signal.

### 3. Tiering & offload
- GPU HBM → CPU DRAM → NVMe.
- Cost/latency curves per tier.
- **FlexKV**, **LMCache**, **Mooncake Store** — tiered KV systems.

### 4. Disaggregation
- Prefill on one pool, decode on another.
- KV transfer cost: RDMA / NVLink / PCIe.
- Mooncake, DistServe, Splitwise.

### 5. Compression & sparsification
- Quantization: FP8 KV, INT4 KV.
- Eviction: StreamingLLM, H2O, Scissorhands.
- Low-rank / shared KV: MLA (DeepSeek), GQA.

## Key metrics

| Metric | Why it matters |
|---|---|
| Cache hit rate | Scheduling + capacity planning |
| Bytes/token | Determines max concurrency at a given HBM |
| Transfer BW utilization | Disagg / offload viability |
| Eviction miss latency | Tail latency under memory pressure |

## Canonical references

- vLLM / PagedAttention (SOSP 2023)
- *SGLang* (2024) — RadixAttention
- *Mooncake* (2024)
- *DistServe* — prefill/decode disaggregation
- *H2O*, *StreamingLLM* — eviction
- *DeepSeek-V2 / V3* — MLA
