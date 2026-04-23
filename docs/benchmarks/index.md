---
title: Benchmarks & Evaluation
tags: [benchmarks, eval]
---

# Benchmarks & Evaluation

You can't optimize what you can't measure. This is the boring but load-bearing section.

## Systems benchmarks

| Benchmark | What it measures |
|---|---|
| **MLPerf Training / Inference** | Vendor-neutral reference |
| **LLM-Perf Leaderboard** (HF) | Throughput / latency on open models |
| **GenAI-Perf** (NVIDIA) | TRT-LLM / Triton serving perf |
| **vLLM bench suite** | vLLM throughput, TTFT, TPOT |
| **Serve-bench / Burstbench** | Request-pattern realism |

## Capability benchmarks

| Category | Benchmarks |
|---|---|
| General knowledge | MMLU, MMLU-Pro, GPQA |
| Math | GSM8K, MATH, AIME, Putnam |
| Code | HumanEval, MBPP, LiveCodeBench, SWE-Bench (Verified) |
| Reasoning | BBH, ARC-AGI, HLE |
| Long context | RULER, LongBench, ∞Bench, NIAH |
| Agents | WebArena, GAIA, OSWorld, AgentBench |
| Multimodal | MMMU, MathVista, Video-MME |
| Safety / refusal | HarmBench, WMDP |

## Evaluation pitfalls

- **Contamination** — benchmark leaked into pretrain.
- **Prompt sensitivity** — 5% swings from phrasing.
- **Compute-equalized eval** often missing for RL-trained models.
- **Sampling settings** — temperature, top-p, maj@k.
- **Long-context needle-in-haystack** overstates real capability.

## How I use benchmarks here

- One-line mention in paper notes when authors report.
- Full write-up only when the benchmark itself is the contribution (e.g. RULER, SWE-Bench).

## Canonical references

- *MLPerf Inference* methodology papers
- *Holistic Evaluation of Language Models (HELM)*
- *RULER* (Hsieh et al., 2024)
