#!/usr/bin/env python3
"""Scaffold a new paper note in docs/papers/posts/.

Usage:
    python scripts/new_paper.py "Paper title" --arxiv 2601.12345 --category attention --tags attention,flash
    python scripts/new_paper.py "Paper title" --category agentic-rl
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
    ap.add_argument("title")
    ap.add_argument("--arxiv", default="")
    ap.add_argument("--venue", default="arxiv")
    ap.add_argument("--category", required=True,
                    help=f"One of: {', '.join(sorted(VALID_CATEGORIES))}")
    ap.add_argument("--tags", default="", help="comma-separated")
    ap.add_argument("--tier", default="L3", choices=["L1", "L2", "L3", "L4", "L5"])
    ap.add_argument("--status", default="reading", choices=["to-read", "reading", "read"])
    ap.add_argument("--date", default="", help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    if args.category not in VALID_CATEGORIES:
        print(f"ERROR: unknown category '{args.category}'.", file=sys.stderr)
        print(f"Valid: {sorted(VALID_CATEGORIES)}", file=sys.stderr)
        return 1

    date = args.date or datetime.date.today().isoformat()
    slug = slugify(args.title)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "docs" / "papers" / "posts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date}-{slug}.md"
    if out_file.exists():
        print(f"ERROR: {out_file} already exists", file=sys.stderr)
        return 1

    frontmatter = [
        "---",
        f"date: {date}",
        "categories:",
        f"  - {args.category}",
        f"tags: [{', '.join(tags)}]" if tags else "tags: []",
        f'arxiv: "{args.arxiv}"',
        f'venue: "{args.venue}"',
        f"tier: {args.tier}",
        f"status: {args.status}",
        "---",
        "",
        f"# {args.title}",
        "",
        "<!-- more -->",
        "",
        "## TL;DR",
        "",
        "## Problem",
        "",
        "## Key ideas",
        "1.",
        "2.",
        "",
        "## Method",
        "",
        "## Experiments",
        "",
        "## My take",
        "",
        "## Open questions",
        "- [ ]",
        "",
    ]
    out_file.write_text("\n".join(frontmatter))
    print(out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
