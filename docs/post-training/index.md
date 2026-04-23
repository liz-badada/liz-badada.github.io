---
title: Post-training (RLHF / DPO / GRPO)
tags: [post-training, rlhf, rl]
---

# Post-training

Everything after pretraining + SFT. Alignment, reasoning uplift, domain adaptation.

## Sub-areas

### Reward modeling
- Pairwise Bradley-Terry reward models.
- Process reward models (PRMs) — per-step scoring.
- Generative / LLM-as-judge rewards.
- Reward hacking / reward model drift.

### Online RL
- **PPO** — classic RLHF.
- **GRPO** (DeepSeekMath/R1) — group-relative, no value net.
- **RLOO**, **REINFORCE++**.
- **DAPO**, **Dr.GRPO** — fixes to GRPO.
- KL control, clip ratios, length bias.

### Offline / contrastive
- **DPO** (Rafailov et al.) — closed-form for BT reward.
- **IPO / SimPO / KTO / ORPO** — DPO variants.
- **Iterative DPO**, **online DPO**.

### Reasoning-focused post-training
- **o1 / R1-style** long chain-of-thought RL.
- Self-consistency, self-critique.
- STaR / Rest-EM / V-STaR — bootstrapped reasoning.

### Systems for RLHF
- **OpenRLHF**, **TRL**, **verl**, **NeMo-Aligner**, **AReaL**.
- Rollout vs training colocation.
- Hybrid engine (train model + rollout model in one process).
- KV reuse across rollouts.

## Canonical references

- *InstructGPT* (Ouyang et al., 2022)
- *Direct Preference Optimization* (Rafailov et al., 2023)
- *DeepSeek-R1* (2025) — GRPO + reasoning RL
- *RLHF Workflow* (Dong et al., 2024)
