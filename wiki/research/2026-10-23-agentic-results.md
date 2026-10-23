---
type: Research
title: Part 2 results — agent runtime vs RAG workflow, LLM vs Laya/Kev decisions
description: Results of the 2026-10-23 OpenRouter run, eight tracks on the Korean eval set, and what they say about the two questions.
tags: [research, results, agentic-rag, hermes, laya, kev]
timestamp: 2026-10-23T06:40:00Z
resource: ../../results/agentic/REPORT.md
---

# Part 2 results (2026-10-23)

The full tables are in [results/agentic/REPORT.md](../../results/agentic/REPORT.md); the raw per-question
JSONL is next to it. The setup is in [PRD.md](../../PRD.md) and
[OpenRouter and subsets](../decisions/2026-10-23-openrouter-and-subsets.md).

Run conditions:

- **Answer model:** OpenRouter `qwen/qwen3.8-27b`, with DeepInfra bf16 pinned and thinking off.
- **Retrieval:** bge-m3, the reranker and OpenSearch on the PGX.
- **Decision models:** Laya multilingual and Kev-4B, on the PGX GPU.
- **Spend:** about $3.3 in all, over the planned ~$2. The overrun is mostly A1 ($1.2), A2 ($0.7) and
  judging ($0.4).

## Architecture: same 120 questions for every track

| Track | Accuracy | doc_qa | multi_doc | doc_calc | Refusal (ooc) | False refusal (doc) | p50 | Input tokens/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0, always hybrid+rerank | 0.575 | 0.93 | 0.55 | 0.33 | 1.00 | 0.15 | 2.2 s | 1.6k |
| W-llm, RAG workflow | 0.800 | 0.77 | 0.60 | 0.40 | 1.00 | 0.22 | 4.7 s | 2.5k |
| A1, Hermes + search tools | **0.850** | 0.93 | **0.80** | **0.53** | **0.72** | 0.02 | 17.5 s | **24.9k** |
| A2, Hermes + RAG layer (W-llm inside) | 0.775 | 0.70 | 0.60 | 0.47 | 0.92 | 0.20 | 15.9 s | 18.4k |

Notes on reading the table:

- Accuracy covers all six intents. B0 scores 0 on calc and chitchat because it always searches.
- With 120 questions the margin is about ±9 pp, so only large gaps count.
- Latencies were measured with 3–4 questions in flight over the network, so read them as relative.

**What the agent does better.** A1 wins on questions that take several steps:

- **multi_doc:** 0.80, against 0.60 for W-llm.
- **doc_calc:** 0.53, against 0.40.
- **doc_qa:** it reformulates and retries. On average it made 1.5 tool calls on doc_qa, 3 on
  multi_doc and 3.5 on doc_calc, often bm25 then hybrid, and it almost never refuses a question it can
  answer (2%).

**What the workflow does better: grounding, cost and predictability.**

- **Refusals.** A1 answered 28% of the questions it should have refused (out_of_corpus). It gave
  opinions ("가장 좋아하는 색은 파란색"), guessed, or answered from its own knowledge with a citation that
  does not support the answer: E=mc² was cited to a chunk about atomic-clock time-dilation
  experiments. W-llm and B0 refused 100%.
- **ACL set.** A1 answered 5 of 15 questions that the user may not see. The workflow tracks answered
  2–6 of 30. The gold chunk was never retrieved in any track, because the filter is injected by the
  tool server. These answers come from other chunks or from the model's own knowledge.
- **Cost and stability.** A1 used 10× the input tokens of W-llm and was 3.7× slower at p50. Its p95
  was 119 s, and 2 runs hit the 10-turn limit while searching for something absent.

**A2 got the agent's overhead without the agent's accuracy.** Its RAG layer was W-llm, and two W-llm
decisions hurt document questions:

1. **Retrieval mode.** The LLM picked bm25 for 96% of doc_qa questions and never picked hybrid. Gold
   hit fell from 0.95 (B0, always hybrid) to 0.87.
2. **Sufficiency gate.** It refused 22% of answerable doc questions.

A2 inherited both flaws (doc_qa 0.70) and added Hermes' cost (18k tokens per question, 16 s). It did
keep the grounding benefit: it refused 92% of unanswerable questions, against 72% for A1. A2 with a
better layer (always hybrid+rerank, calibrated gate) was not run; see next steps.

## Decision models: Laya and Kev in place of LLM decisions (300 questions)

| | W-llm (Qwen) | W-kev (Kev-4B) | W-laya (Laya) | W-cascade-kev τ=0.5 | W-cascade-laya τ=0.7 |
| --- | --- | --- | --- | --- | --- |
| End-to-end accuracy | **0.783** | 0.657 | 0.357 | 0.663 | 0.657 |
| Intent accuracy / macro-F1 | 0.787 / 0.742 | 0.783 / **0.794** | 0.21 / 0.21 | 0.81 / 0.82 | 0.65 / 0.63 |
| False refusal on doc questions | 0.25 | 0.52 | 0.44 | 0.51 | 0.31 |
| LLM calls / input tokens per question | 3.2 / 2.4k | **1.0 / 0.9k** | 1.0 / 1.0k | 1.8 / 0.8k | 2.4 / 1.5k |
| Decision latency, p50 | ~0.85 s | ~0.25–0.7 s | **~0.05 s** | | |
| Intent ECE | – | 0.21 | 0.34 | | |

### Tool selection by number of tools (4 RAG tools, plus up to 16 distractor business tools)

| Decider | 4 tools | 10 tools | 20 tools | p50 |
| --- | --- | --- | --- | --- |
| Qwen, labels as JSON | **0.957** | **0.927** | **0.929** | 0.8 s |
| Qwen native tool calling | 0.833 (0.887*) | 0.839 (0.903*) | 0.792 (0.847*) | 2.3–3.3 s |
| Kev-4B | 0.807 | 0.812 | 0.821 | 0.22–0.28 s |
| Laya | 0.497 | 0.406 | 0.255 | 0.05 s |

\* Counting a direct refusal without a tool call as correct for `refuse`.

Native tool calling also skipped the calculator on 21 of 40 arithmetic questions: it computed the
answer itself.

- **Kev-4B matches the LLM at choosing intents and tools.** Its intent macro-F1 is 0.79, against 0.74,
  and its tool accuracy holds at about 0.82 from 4 to 20 tools, roughly level with native tool
  calling. It does this with one LLM call per question instead of 3.2, and 64% fewer input tokens.
  The cascade (Kev when p ≥ 0.5, else the LLM) reaches 0.81 intent accuracy while Kev handles 68% of
  questions.
- **Kev loses end to end because of the yes/no sufficiency decision.** Its probability of "yes" is
  low even when the gold passage is retrieved, so it refuses 52% of answerable doc questions. Routing
  is not the problem; the binary gate is uncalibrated.
- **Laya does not work for Korean routing.** Its intent accuracy is 0.21; most document questions
  were labelled out_of_corpus. Tool accuracy falls from 0.50 to 0.26 as tools are added, and it is
  overconfident (ECE 0.34). It is fast (about 50 ms), but that does not help here.
- **The baseline to beat is the LLM as a classifier, not native tool calling.** Asking Qwen to pick
  a label in JSON was the most accurate decider (0.93–0.96) and 3–4× faster than native tool calling.

## Reliability of these numbers

- **The judge holds up.** Claude Haiku 5.5 re-judged 43 Qwen-judged answers and agreed on correctness
  43 of 43 times. Faithfulness agreed 77% of the time.
- **Single run.** Consistency across 3 repeats was 0.85–1.0, lowest for A2 at 0.85.
- **Eval-set caveats:**
  - The reference answers were written by an LLM.
  - Some "out of corpus" entity questions are about famous facts (E=mc², Kant), which tempts any
    model with world knowledge.
  - doc_calc pairs are sometimes only loosely related.
- **τ is optimistic.** It was picked on the same data as the cascade curve.
