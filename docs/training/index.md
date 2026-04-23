---
title: Pretraining & SFT Infra
tags: [training, pretraining, sft]
---

# Pretraining & SFT Infra

The plumbing for dense / MoE pretraining and supervised fine-tuning.

## Sub-areas

### Frameworks
- **Megatron-LM / Megatron-Core** — NVIDIA flagship.
- **DeepSpeed** — ZeRO, Ulysses, training engine.
- **PyTorch FSDP / DTensor** — native.
- **Torchtitan** — reference PyTorch-native trainer.
- **NeMo-Framework** — NVIDIA productized Megatron + data pipelines.
- **Colossal-AI**, **OpenRLHF**, **veScale**.

### Optimizers
- **AdamW** (workhorse).
- **Lion**, **Sophia**, **Shampoo / Distributed Shampoo**.
- **Muon** (2024) — momentum orthogonalized.
- Optimizer-state sharding (ZeRO-1/2/3).

### Data pipelines
- Tokenization at scale.
- Streaming, packed samples, document masking.
- **Mosaic Streaming**, **Nemotron Data Curator**.
- Data mixing (DoReMi, RegMix).

### Checkpointing
- Async, hierarchical.
- **Torch DCP**, **Gemini**.
- Resharding checkpoints across parallelism configs.

### SFT
- Packing, loss masking on instructions.
- Multi-turn chat templates.
- LoRA / QLoRA for parameter-efficient SFT.

## Canonical references

- *Megatron-LM* (Shoeybi et al., 2019)
- *ZeRO* / *ZeRO-Infinity* (DeepSpeed)
- *FSDP* (Zhao et al., 2023)
- *The LLaMA 3 Herd of Models* (Meta, 2024) — pretraining recipe
