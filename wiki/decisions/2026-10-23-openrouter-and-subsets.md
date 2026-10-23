---
type: Decision
title: Answer LLM on OpenRouter, Hermes tracks on a 120-question subset
description: Why the Part 2 runs moved from the PGX Qwen to OpenRouter's qwen/qwen3.8-27b, and why A1/A2 run on a smaller stratified subset (budget about $2).
tags: [decision, openrouter, cost, eval-set]
timestamp: 2026-10-23T04:30:00Z
resource: ../../rag_bench/agentic/evalset.py
---

# Answer LLM on OpenRouter, Hermes tracks on a subset

## The local Qwen was too slow

Six tracks plus the tool sweep ran against Qwen3.8-27B-NVFP4 on the PGX. vLLM reported about 17
generated tokens/s in total with 7 requests in flight, a prompt throughput of 200–1,100 tokens/s and
a prefix-cache hit rate of about 6% (it is a hybrid Mamba model). Hermes questions cost about 70 s
of machine time each, so A1 and A2 alone would have taken about 19 hours.

The owner had allowed OpenRouter if speed was a problem, so the switch was made:

- **Model:** `qwen/qwen3.8-27b`, the same model family and size.
- **Provider:** pinned to DeepInfra (bf16) with `allow_fallbacks: false`, so every call hits one
  backend.
- **Thinking:** off (`reasoning.enabled: false`). The proxy injects this, plus the model id, into
  every request, including Hermes' requests.
- **Re-run:** by the rule in the PRD, every track is re-run on OpenRouter. The partial local results
  were moved out of the repo and are not compared.
- **Unchanged:** retrieval (bge-m3, reranker, OpenSearch) and the decision models (Laya, Kev-4B) stay
  on the PGX.

## Budget about $2, plan B

The owner asked for about $2. Measured cost per question (DeepInfra pricing):

| Track | Input tokens per question | Cost per question |
| --- | --- | --- |
| A1 | ~20k | ~0.35 ¢ |
| A2 | ~8k | ~0.15 ¢ |
| W-llm | ~3k | ~0.06 ¢ |
| B0, W-laya, W-kev | 1–1.5k | ~0.03 ¢ |

So the cheap tracks run the full set and only Hermes is cut back:

- **B0 and the W tracks:** the full main set (300) and ACL set (30).
- **A1 and A2:** the `small` subset, which is 120 main questions plus 15 ACL questions, stratified by
  intent and gold tool:
  - doc_qa 30, multi_doc 20, doc_calc 15, out_of_corpus 25, chitchat 15, calc 15
- **Consistency:** 20 questions from the small subset, run 3 times by every track.
- **Architecture comparison (A vs W):** made on the same 120 questions. The decision-model comparison
  (W-laya and W-kev against W-llm) uses all 300.
- **Estimated total:** about $2.3, at most about $3.

## Consequences

- With 120 questions, overall accuracy has a margin of about ±9 pp, and more per intent. The PRD's
  3 pp thresholds can't be decided for A1/A2; only large gaps can. The 300-question W comparison is
  firmer.
- The runs use `--workers` (3–8 questions in flight) for speed. Latency is therefore measured under
  load and over the network, and reads as relative, not absolute. The decision models' own latency is
  timed per call and is not affected.
