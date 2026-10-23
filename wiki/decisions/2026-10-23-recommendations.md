---
type: Decision
title: Recommendations — agent runtime + RAG layer, and Laya/Kev for tool selection
description: Verdicts on the owner's two questions from the 2026-10-23 run, with the evidence behind them and what would change them.
tags: [decision, recommendation, agentic-rag, hermes, laya, kev]
timestamp: 2026-10-23T06:40:00Z
resource: ../research/2026-10-23-agentic-results.md
---

# Recommendations (2026-10-23)

Evidence: [Part 2 results](../research/2026-10-23-agentic-results.md). Scope limits: one 1,000-article
Korean Wikipedia corpus, one 27B answer model, one run. Gaps under about 9 pp on the 120-question
architecture set are not significant.

## 1. Agent runtime with a RAG layer: yes, but the layer must own grounding and retrieval

**Verdict: the direction is right, but only with a strong RAG layer. This run did not show A2 beating
the alternatives (H3 not met).**

- **The agent handles multi-step questions better.** A1 led on multi_doc (0.80 against 0.60) and
  doc_calc (0.53 against 0.40). A fixed workflow needs extra branches for every new question shape.
- **A bare agent with search tools is hard to control for an enterprise RAG product.**
  - It failed to refuse 5 of 25 unanswerable questions, where the workflow refused all of them. Two
    were persona answers to opinion questions. One was a fact from the model's own knowledge with a
    citation that does not support it (E=mc²).
  - It used 10× the tokens of the workflow, ran 3.7× slower at p50 and had a p95 near 2 minutes. It
    sometimes looped until the turn limit.
  - Its multi-step gains rest on a few questions (4 of 20 and 2 of 15), so they are directional.
- **A2 shows that the layer decides the quality.** A2 was the proposed shape. It kept most of the
  grounding (it refused 92% of unanswerable questions, against 80% for A1), but it inherited the
  layer's flaws:
  - LLM mode routing chose bm25 96% of the time.
  - An over-strict sufficiency gate made it refuse answerable questions.

  It also paid the agent's overhead (18k tokens per question).

**How to build it.**

- The agent runtime plans and composes: it splits questions, calls the calculator and other business
  tools, and merges answers.
- The RAG layer is the only path to documents. It always uses hybrid search plus reranking, which
  gave the best gold hit (0.95), so no LLM picks the retrieval mode. Inside the layer it:
  - applies the access filter server-side
  - returns passages with citations
  - enforces refusal with a calibrated threshold, such as a reranker score, not an LLM yes/no
- Keep the agent's own system prompt short, and give it the RAG-layer tool alongside the business
  tools, not raw search tools.

**What would confirm this.** Re-run A2 with the improved layer (always hybrid+rerank, gate on the
reranker score). It should keep A1's multi-step gains and the workflow's refusal and ACL behaviour.
The cost would be about $0.8.

## 2. Laya and Kev in place of LLM tool selection

By the PRD's rules (H4–H7), **neither is a GO as configured.**

### Laya: NO-GO

- **Intent:** accuracy 0.21.
- **Tool choice:** 0.50 with 4 tools, falling to 0.26 with 20.
- **Calibration:** overconfident (ECE 0.34).
- **Cascade:** it only helps when the LLM handles most questions anyway.
- **Speed:** about 50 ms, which does not make up for this in a Korean product.

### Kev-4B: NO-GO as configured; routing-only use looks promising, pending a re-test

By the pre-registered rules Kev fails both GO (H4 end to end, H5 latency) and conditional GO (H6: ECE
0.21 against ≤0.1, 68% handled against ≥70%, cascade end to end 0.66 against 0.78). The routing-only
reading below was formed after seeing the results, so it needs the re-test before it counts.

What works:

- **Routing matches the LLM.** Intent macro-F1 is 0.79 against 0.74 for the LLM (H4's routing half is
  met).
- **Tool choice holds as tools grow.** It stays at about 0.82 from 4 to 20 tools, level with Qwen's
  native tool calling (0.79–0.84). It cuts LLM calls from 3.2 to 1.0 and input tokens by 64%.
- **The cascade works for intents.** With Kev at p ≥ 0.5 and the LLM below, intent accuracy is 0.81
  and Kev handles 68% of questions. This is close to the H6 target (≥70%), but ECE is 0.21 against a
  target of ≤0.1.

What fails:

- **End-to-end accuracy.** It is 0.66 against 0.78 (H4 not met). There are 45 doc questions W-llm got
  right and W-kev got wrong. The yes/no gate caused 27 of them (60%); Kev refuses 52% of answerable doc
  questions. Routing caused the other 17: 10 sent to out_of_corpus, 7 with the wrong document intent.
  Fixing the gate alone would therefore not close the whole gap.
- **The latency target.** At 0.22–0.28 s it misses H5's 50 ms, though it is still 3× faster than the
  LLM as a classifier and about 10× faster than native tool calling.

**Product advice, if the re-test confirms it.**

- Use Kev-4B as the first-stage intent and tool router, with the LLM as a fallback below a confidence
  threshold. Calibrate the threshold per deployment on a few hundred labelled questions.
- Do not use Kev or Laya for answerability gates. Use reranker scores, or an LLM, for those.
- Before shipping, compare against the cheaper baseline found here. Qwen as a JSON label classifier
  reached 0.93–0.96 tool accuracy in about 0.8 s, so for products already paying for an LLM call the
  case for Kev is mainly cost and GPU footprint, not accuracy.
- Budget for one more model to serve: a 4B model on GPU with its own server. The LoRA plus pointer
  head cannot be served by vLLM.

**What would change the verdict.**

- Re-run W-kev with a calibrated sufficiency threshold, or the reranker gate. If end-to-end accuracy
  then comes within 3 pp of W-llm, Kev becomes a GO for routing.
- A larger labelled set would settle ECE and τ without the optimistic same-data choice.
