---
hide:
  - navigation
  - toc
---

# AI Infra Notes

Notes on **LLM inference**, **training**, **RL**, **agents**, **multimodal**, and **GPU systems** — built while reading papers and open-source projects.

---

## Browse by area

<div class="cluster-grid" markdown>

<div class="cluster-card cluster-inference" markdown>
### :material-speedometer: Inference
<p class="cluster-desc">The runtime path: engines, KV cache, kernels, scheduling, quantization.</p>
<div class="cluster-links" markdown>
[Engines](engines/index.md) ·
[KV Cache](kv-cache/index.md) ·
[Attention](attention/index.md) ·
[Scheduling](scheduling/index.md) ·
[Quantization](quantization/index.md)
</div>
</div>

<div class="cluster-card cluster-parallelism" markdown>
### :material-source-branch: Parallelism & Communication
<p class="cluster-desc">How compute gets split and how devices talk.</p>
<div class="cluster-links" markdown>
[Parallelism](parallelism/index.md) ·
[Ultra-scale](large-scale/index.md) ·
[Comm & Networking](communication/index.md) ·
[MoE](moe/index.md) ·
[Long Context](long-context/index.md)
</div>
</div>

<div class="cluster-card cluster-training" markdown>
### :material-school: Training & RL
<p class="cluster-desc">Pretraining, post-training, and RL on agents.</p>
<div class="cluster-links" markdown>
[Pretraining & SFT](training/index.md) ·
[Post-training](post-training/index.md) ·
[Agentic RL](agentic-rl/index.md)
</div>
</div>

<div class="cluster-card cluster-applications" markdown>
### :material-puzzle: Applications
<p class="cluster-desc">Where infra gets used.</p>
<div class="cluster-links" markdown>
[Agents & RAG](agents/index.md) ·
[Multimodal & Omni](multimodal/index.md) ·
[Diffusion & Gen Media](diffusion/index.md)
</div>
</div>

<div class="cluster-card cluster-foundations" markdown>
### :material-chip: Foundations
<p class="cluster-desc">Silicon, modeling, and measurement.</p>
<div class="cluster-links" markdown>
[Hardware](hardware/index.md) ·
[Perf Modeling](perf-modeling/index.md) ·
[Benchmarks](benchmarks/index.md)
</div>
</div>

<div class="cluster-card" style="--c: var(--md-default-fg-color--lighter);" markdown>
### :material-notebook: Latest
<p class="cluster-desc">Dated reading notes and project deep-dives.</p>
<div class="cluster-links" markdown>
[All papers →](papers/index.md) ·
[All projects →](projects/index.md) ·
[Browse tags →](tags.md)
</div>
</div>

</div>

---

See [the full roadmap →](roadmap.md) for a single-page map of how these areas connect.
