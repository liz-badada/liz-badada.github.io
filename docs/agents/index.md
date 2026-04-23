---
title: Agents & RAG
tags: [agents, rag]
---

# Agents & RAG

The layer between users and models. Where context engineering, tool use, and memory live.

## Sub-areas

### Agent patterns
- **ReAct** — reason + act loop.
- **Reflexion** — self-critique on failure.
- **Plan-and-Execute** / **Plan-Revise-Act**.
- **Code agent** (Voyager, SWE-agent, OpenDevin).
- **Deep research / report writing**.

### Tool use
- Function calling formats (OpenAI, Anthropic tools, XML).
- Tool schema design, error surfaces.
- MCP (Model Context Protocol) — standardizing tool exposure.

### Memory
- Short-term: conversation buffer.
- Long-term: vector stores, summaries, knowledge graphs.
- **MemGPT**, **Letta**, **Mem0**.

### Multi-agent
- Supervisor / worker patterns.
- **AutoGen**, **CrewAI**, **LangGraph**.
- Communication protocols, role specialization.

### RAG
- Retrieval: BM25 + dense + hybrid, reranking.
- Chunking, query rewriting, HyDE.
- GraphRAG, self-RAG, corrective RAG.
- Eval: Ragas, ARES, groundedness metrics.

### Agent infra
- Trajectory logging / replay.
- Guardrails, safety filters.
- Cost tracking and budgeting.
- **LangSmith**, **Arize Phoenix**, **Langfuse**.

## Canonical references

- *ReAct* (Yao et al., 2022)
- *Reflexion* (Shinn et al., 2023)
- *Voyager* (Wang et al., 2023)
- *SWE-agent* (Yang et al., 2024)
- *MemGPT* (Packer et al., 2023)
