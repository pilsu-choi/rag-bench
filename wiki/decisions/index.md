---
type: Index
title: Decisions
description: "Choices made for the benchmark and why."
tags: [index, decisions]
timestamp: 2026-10-23T02:35:00Z
---

# Decisions

- [Benchmark design](2026-10-20-benchmark-design.md): two parts, same vectors, equal-recall comparison, one container at a time.
- [Model choice](2026-10-20-model-choice.md): bge-m3 and bge-reranker-v2-m3 on vLLM, with Qwen3.8-27B for query synthesis.
- [Part 2 design](2026-10-23-agentic-rag-design.md): agent runtime vs RAG workflow, LLM tool calling vs Laya/Kev, six tracks on one stack.
- [OpenRouter and subsets](2026-10-23-openrouter-and-subsets.md): local Qwen too slow, so qwen/qwen3.8-27b on OpenRouter (DeepInfra pinned); Hermes tracks on a 120-question subset to stay near $2.
- [Recommendations](2026-10-23-recommendations.md): agent runtime + RAG layer is right only if the layer owns retrieval and grounding; Laya NO-GO; Kev-4B conditional GO as a first-stage router, not for answerability gates.
