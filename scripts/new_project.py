#!/usr/bin/env python3
"""Scaffold a new project deep-dive in docs/projects/.

Usage:
    python scripts/new_project.py vllm --category engines --repo https://github.com/vllm-project/vllm
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

VALID_CATEGORIES = {
    "engines", "kv-cache", "attention", "scheduling", "quantization",
    "parallelism", "large-scale", "communication", "moe", "long-context",
    "training", "post-training", "agentic-rl",
    "agents", "multimodal", "diffusion",
    "hardware", "perf-modeling", "benchmarks",
}


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--category", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--org", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--tier", default="L3", choices=["L1", "L2", "L3", "L4", "L5"])
    args = ap.parse_args()

    if args.category not in VALID_CATEGORIES:
        print(f"ERROR: unknown category '{args.category}'.", file=sys.stderr)
        return 1

    slug = slugify(args.name)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    date = datetime.date.today().isoformat()

    root = Path(__file__).resolve().parent.parent
    out_file = root / "docs" / "projects" / f"{slug}.md"
    if out_file.exists():
        print(f"ERROR: {out_file} already exists", file=sys.stderr)
        return 1

    frontmatter = [
        "---",
        f"title: {args.name}",
        f"date: {date}",
        f'repo: "{args.repo}"',
        f'org: "{args.org}"',
        f"tags: [{', '.join(tags)}]" if tags else "tags: []",
        "categories:",
        f"  - {args.category}",
        f"tier: {args.tier}",
        "status: active",
        "---",
        "",
        f"# {args.name}",
        "",
        "## TL;DR",
        "",
        "## Architecture overview",
        "```mermaid",
        "flowchart TB",
        "    A --> B",
        "```",
        "",
        "## Key components",
        "",
        "## Notable design choices",
        "",
        "## Performance notes",
        "",
        "## Personal impressions",
        "",
    ]
    out_file.write_text("\n".join(frontmatter))
    print(out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
