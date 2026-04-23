---
title: Multimodal & Omni
tags: [multimodal, vlm, omni]
---

# Multimodal & Omni

From VLM (vision-language) to any-to-any. Infra implications go beyond adding a vision tower.

## Sub-areas

### VLM architecture
- **CLIP-style encoder + LLM** (LLaVA, InternVL).
- **Native multimodal** (Chameleon, Gemini, GPT-4o).
- **Dynamic resolution / patch tiling** (Qwen2-VL, InternVL2, NaViT).
- **Perceiver / resampler** vs linear projector.

### Omni models
- **Qwen2.5-Omni**, **MiniCPM-o**, **Gemini 2.0 Flash**, **GPT-4o**.
- Any-to-any: text / image / audio / video in and out.
- Talker-thinker architectures.
- Streaming duplex voice.

### Video
- Temporal tokenization — how many tokens / second.
- Key-frame vs uniform sampling.
- Long video: memory bank, causal caching.

### Audio & speech
- Whisper / Canary / Seed-ASR.
- Neural codecs (EnCodec, SNAC, Mimi).
- Real-time TTS (VALL-E, Parler, F5-TTS).

### Inference implications
- Prefill cost dominated by vision tokens (hundreds of images → 100k+ tokens).
- KV cache: huge for video, still mostly dense.
- Encoder caching / reuse across multi-turn.
- Streaming audio input = no end-of-sequence.

### Training
- Staged training (vision encoder → LLM → joint).
- Curriculum: single-image → multi-image → interleaved → video.
- Balancing modalities to avoid regression.

## Canonical references

- *LLaVA* (Liu et al., 2023)
- *Flamingo* (Alayrac et al., 2022)
- *Chameleon* (Meta, 2024)
- *GPT-4o* / *Gemini* system reports
- *Qwen2.5-Omni* technical report
