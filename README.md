# AI Infra Knowledge Base

Personal knowledge base on LLM inference, training infra, RL, agents, multimodal, and GPU systems. Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Structure

```
docs/
├── index.md             # home
├── roadmap.md           # full 19-topic map
├── papers/posts/        # dated reading notes (blog-style, auto-archived)
├── projects/            # deep-dives on open-source codebases
├── templates/           # paper.md + project.md note templates
├── reading-list.md      # queue
└── <19 topic dirs>/     # engines, kv-cache, attention, ...
scripts/
├── new_paper.py         # scaffold a new paper note
└── new_project.py       # scaffold a new project deep-dive
```

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve                    # http://127.0.0.1:8000
```

## Adding a paper note

```bash
python scripts/new_paper.py "FlashAttention-3" \
    --arxiv 2407.08608 \
    --category attention \
    --tags "flash,attention,hopper" \
    --tier L4
```

This writes `docs/papers/posts/YYYY-MM-DD-flashattention-3.md` with the right frontmatter.

## Adding a project deep-dive

```bash
python scripts/new_project.py vllm \
    --category engines \
    --repo https://github.com/vllm-project/vllm \
    --org vllm-project
```

## Deploy to GitHub Pages

1. Create repo `liz-badada.github.io` on GitHub (public, no init).
2. From this directory:
   ```bash
   git init
   git remote add origin git@github.com:liz-badada/liz-badada.github.io.git
   git add .
   git commit -m "Initial KB scaffold"
   git branch -M main
   git push -u origin main
   ```
3. The workflow in `.github/workflows/deploy.yml` will build and push to `gh-pages` branch.
4. Repo → Settings → Pages → Source: `gh-pages` branch, `/` root.
5. Site will be live at `https://liz-badada.github.io` in a minute or two.

## Taxonomy (19 topics, 5 clusters)

**Inference** — engines · kv-cache · attention · scheduling · quantization
**Parallelism & Comm** — parallelism · large-scale · communication · moe · long-context
**Training & RL** — training · post-training · agentic-rl
**Applications** — agents · multimodal · diffusion
**Foundations** — hardware · perf-modeling · benchmarks

See `docs/roadmap.md` for the full mental map.

## Reading tiers

| Tier | Meaning |
|---|---|
| L1 | Aware it exists |
| L2 | Abstract + intro + conclusion |
| L3 | Full read + key experiments |
| L4 | Can reproduce key figures |
| L5 | Applied to own work |
