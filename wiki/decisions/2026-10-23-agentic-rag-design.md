---
type: Decision
title: Part 2 design — agent runtime vs RAG workflow, LLM tool calling vs decision models
description: Why Part 2 compares six tracks on one stack, what each one isolates, and how tool choice is scored without gold labels.
tags: [decision, agentic-rag, hermes, laya, kev, tool-calling]
timestamp: 2026-10-23T02:35:00Z
resource: ../../PRD.md
---

# Part 2 design

The full spec is in [PRD.md](../../PRD.md). This page records the reasoning behind it.

## Two questions from the owner

1. **Architecture.** For an enterprise RAG product, which works better: a general agent runtime
   (Hermes Agent) with retrieval tools, or a RAG-specific workflow? And is "agent runtime with a RAG
   layer on top" the right shape?
2. **Tool calling.** Can a decision model (Laya, Kev) replace the tool-choice half of LLM tool calling
   well enough that adding it to a commercial product pays off?

## Tracks and what each one isolates

| Track | Isolates |
| --- | --- |
| B0, fixed hybrid+rerank, then answer | The floor: no decisions at all |
| W-llm, RAG workflow with Qwen decisions | The workflow itself, with an LLM deciding |
| W-laya / W-kev, same workflow with decision-model decisions | Decision maker only: W-llm against W-laya/W-kev |
| W-cascade, decision model with LLM fallback below a threshold τ | Whether the probabilities are trustworthy enough for a product |
| A1, Hermes with raw retrieval tools | General agent against workflow: A1 against W |
| A2, Hermes with one `ask_knowledge_base` tool that wraps the best W | The proposed enterprise shape: A2 against A1 and W |

All tracks share the OpenSearch index on the PGX, bge-m3, bge-reranker-v2-m3 and Qwen3.8-27B-NVFP4
(thinking off, temperature 0) as the answer model, so only the decision maker or the architecture
changes.

## Choices and reasons

- **OpenSearch on the PGX, not FAISS in-process.** One engine offers BM25, dense and hybrid search.
  That turns "which retrieval tool" into a real choice for the router. It also frees the 15 GB WSL
  box. In Part 1 OpenSearch used 3.9–5.5 GB.
- **Base scale only (9,069 chunks).** Part 1 showed that the distractors change retrieval scores, not
  the decision behaviour this part is about.
- **No gold label for bm25, dense or hybrid.** The router's retrieval choice is scored by outcome,
  against a per-query oracle (best of the three) and against always-hybrid. Hybrid+rerank already
  leads on quality (base nDCG@10: 0.904, against 0.82 for BM25 and 0.79 for dense). So the hypothesis
  is equal quality at lower latency, not higher quality.
- **Decision models replace tool selection only.** They choose a label (with a no-tool option) and
  cannot write arguments. Search uses the raw question. Calculator expressions and per-entity queries
  need one LLM call, which is counted separately.
- **Six intents with fixed expected behaviour.** The intents are `doc_qa`, `multi_doc`, `doc_calc`,
  `out_of_corpus`, `chitchat` and `calc`. `out_of_corpus` must answer "문서에서 찾을 수 없습니다", which
  makes refusal measurable. A question alone often cannot tell whether the corpus covers it, so the
  workflow adds a sufficiency yes/no after retrieval. `doc_calc` is in the set because it is where an
  agent should beat a fixed workflow.
- **Tool-count sweep (5/10/20 tools, with distractor business tools).** Commercial products expose many
  tools, and Laya is reported weak with many labels (0.425 on Banking77, 77 labels).
- **Operational metrics next to quality.** Each track is measured on run-to-run consistency (3 runs on
  50 questions), ACL leakage (30 questions answerable only from another tenant), trace completeness,
  latency and tokens. These decide the architecture question as much as accuracy does.
- **Access filter injected by the tool server.** The model never sees the tenant as an argument. A side
  experiment measures leakage when the filter is only stated in the prompt.
- **One logging proxy for every LLM call.** This counts calls and tokens the same way for Hermes and
  for the in-house harness.
- **Qwen as judge, cross-checked.** Fifty answers are re-judged by a different model on OpenRouter to
  estimate self-preference bias.

## Verdicts this produces

- Architecture: H1–H3 in the PRD.
- Decision models: GO if they match W-llm on their own (H4, H5). Conditional GO if they only work as a
  cascade in front of the LLM (H6). NO-GO otherwise.
