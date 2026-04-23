---
title: Agentic RL
tags: [agentic-rl, rl, agents]
---

# Agentic RL

RL where the policy is an LLM that takes multi-step actions in a tool-using environment. The fastest-moving corner of post-training in 2025–2026.

## What makes it different from RLHF

| RLHF | Agentic RL |
|---|---|
| Single-turn preference | Multi-turn trajectory |
| Reward from human / RM | Reward from env / verifier |
| Short rollouts | Long-horizon, tool calls, search |
| Context fits easily | Context grows with trajectory |

## Sub-areas

### Environments
- **Code execution** (sandboxes, SWE-Bench, LiveCodeBench).
- **Web navigation** (WebArena, VisualWebArena, Mind2Web).
- **Math / theorem proving** (Lean, MiniF2F).
- **Scientific tools** (search + calculator + code).
- **Game envs** — strategy, long-horizon.

### Algorithms
- **GRPO over trajectories**.
- **RLOO**, **REINFORCE with baselines**.
- **Outcome vs process rewards** on traces.
- **Tree search + RL** (rStar-Math, MCTS-augmented).

### Credit assignment
- Per-step vs per-trajectory reward.
- Advantage shaping for long horizons.
- Value function estimation on token sequences.

### Systems for agentic RL
- **verl** (ByteDance) — hybrid engine, veRL.
- **AReaL**, **RAGEN**, **Agent Lightning**.
- Rollout pool management — async vs sync, env-side batching.
- KV reuse across tool-call turns (prefix caching essential).
- Env sandboxing (Firecracker, Docker pools).

### Stability issues
- Reward sparsity → exploration failure.
- Long rollouts → memory pressure during training.
- Tool-call format drift mid-training.

## Canonical references

- *DeepSeek-R1* — GRPO on reasoning
- *rStar-Math* (Microsoft, 2025)
- *verl: HybridFlow* (ByteDance, 2024)
- *Agent Q*, *ReAct*, *Reflexion* (foundational)
