---
title: Diffusion & Gen Media
tags: [diffusion, video-gen, image-gen]
---

# Diffusion & Gen Media

Image, video, and audio generation. Compute profile differs sharply from autoregressive LLMs (no KV cache, lots of attention over long sequences of latents).

## Sub-areas

### Architectures
- **U-Net** — SD 1.x / 2.x legacy.
- **DiT** (Diffusion Transformer) — Sora, SD3, Flux.
- **MM-DiT** — joint text + image attention (Stable Diffusion 3).
- **Rectified flow / flow matching** — SD3, Flux.

### Video generation
- **Sora**, **Veo 2**, **Kling**, **Wan 2.1**, **Hunyuan Video**, **Mochi-1**, **CogVideoX**.
- 3D VAE for spatiotemporal compression.
- Causal VAE for streaming.

### Step distillation
- **LCM** (Latent Consistency Models).
- **SDXL-Turbo**, **SD3-Turbo**.
- **DMD / DMD2** — distribution matching distillation.
- 1–4 step inference from 50-step teachers.

### Serving optimizations
- **Feature caching** — reuse attention / MLP outputs across steps (**DeepCache**, **TGATE**, **Block Caching**).
- **Quantization** — INT8/FP8 for diffusion transformers.
- **Token merging / pruning** (ToMeSD).
- **Parallel sampling** — classifier-free guidance split.

### Audio generation
- **Stable Audio**, **AudioLDM**, **MusicGen**.
- Diffusion + neural codec.

### Eval
- FID / FVD / CLIPScore (image).
- VBench (video).
- User studies still dominate.

## Canonical references

- *Scalable Diffusion Models with Transformers* (Peebles & Xie, 2022) — DiT
- *Rectified Flow* (Liu et al., 2022)
- *Sora technical report* (OpenAI, 2024)
- *DeepCache* (Ma et al., 2024)
