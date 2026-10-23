---
type: Log
title: Work log
description: Chronological history of the rag-bench work, newest last.
tags: [log, history]
timestamp: 2026-10-20T09:15:00Z
---

# Work log

Times are UTC. Each entry links to the page with the details.

## 2026-10-20

- **06:50** The goal was to compare OpenSearch, Milvus, FAISS and pgvector for RAG retrieval. The owner
  assumed about 1,000 Korean documents and no existing qrels, so the qrels had to be synthesised.
  Agreed on a two-part design: an ANN engine comparison against exact kNN, and a retrieval
  pipeline comparison against qrels ([design](decisions/2026-10-20-benchmark-design.md)).
- **06:58** `prep.py` built the corpus: 1,000 Korean Wikipedia articles (9,064 chunks at that point)
  plus 200,000 distractor chunks.
- **07:02** `synth.py` generated 600 questions with Qwen3.8-27B on the PGX in 3.5 min, at 8 concurrent
  requests with thinking off. 599 passed the filter.
- **07:04** Downloaded bge-m3 and bge-reranker-v2-m3. The first try failed because of how `hf download`
  parses `--exclude` ([lesson](lessons/2026-10-20-hf-download-flags.md)). Stopped Qwen and started both
  models with vLLM `--runner pooling` ([model choice](decisions/2026-10-20-model-choice.md)).
- **07:05** Three parallel sub-agents wrote the pgvector, Milvus and OpenSearch adapters while the
  orchestrator wrote the FAISS adapter, the shared code and `bench.py`. All three adapters passed their
  smoke tests.
- **07:21** Embedding the distractors failed on a 94k-character chunk. Fixed the chunking to hard-split
  run-on sentences, dropped the one query whose source chunk changed (598 left) and re-embedded
  everything (done 07:43; [lesson](lessons/2026-10-20-run-on-chunks.md)).
- **07:35** A trial pgvector run showed 23 ms at ef 32 and 64 against 1–3 ms at the other ef values.
  The planner had chosen an exact sort over the HNSW index
  ([lesson](lessons/2026-10-20-pgvector-planner-seqscan.md)).
- **07:44–08:14** `scripts/run-all.sh` ran every store at both scales, then measured model latency.
- **08:15–08:30** At 209k vectors Milvus reached only 224 QPS with 8 clients. It had sealed the data
  into 10 segments, and compaction did not merge them. Raised the seal size in
  `docker/milvus-user.yaml`, which gave 2 segments and 777 QPS, and re-ran Milvus at both scales
  ([lesson](lessons/2026-10-20-milvus-segment-sealing.md)).
- **08:35** Generated `results/REPORT.md` and the recall-vs-latency chart
  ([results](research/2026-10-20-results.md)). The chart showed that the warm-up was too short, so
  `bench.py` now warms up with every query. This run was not repeated ([issues](issues/index.md)).
- **09:15** Moved the model host address out of the code and into `PGX_HOST`, wrote this wiki,
  created the GitHub repo and pushed `dev` ([runbook](runbooks/2026-10-20-run-benchmark.md)).

## 2026-10-23

- **01:50** Started Part 2: agentic RAG evaluation. The owner wants answers to two questions. First,
  general agent runtime (Hermes Agent) with retrieval tools vs a RAG-specific workflow, and whether
  "agent runtime with a RAG layer" fits an enterprise product. Second, whether decision models (Laya,
  Kev) can replace LLM tool calling for tool selection well enough to ship. Agreed with the owner:
  - keep the work in this repo
  - use OpenSearch on the PGX as the only store
  - answer with Qwen3.8-27B-NVFP4 (OpenRouter allowed if too slow)
  - use the Part 1 Korean corpus
- **02:30** Wrote [PRD.md](../PRD.md), [AGENTS.md](../AGENTS.md) and the
  [Part 2 design](decisions/2026-10-23-agentic-rag-design.md). Three sub-agents started in parallel:
  - bringing up the PGX stack (Qwen with tool calling, embed, rerank, OpenSearch)
  - a Hermes headless and isolation check
  - a Laya and Kev check
- **03:10** The Laya and Kev check was done on CPU with 6 questions (directional only). Kev
  (`jaredpalmer/kev-*`, a LoRA plus a pointer head on Qwen3.5-Base, with its own server) got 6/6
  intents. Laya (`convaiinnovations/laya`, multilingual checkpoint) got 3/6 and was overconfident when
  wrong. Neither model can be served by vLLM.
- **03:20** The Hermes check passed on a mock LLM: it runs headless, with isolated state, stream-json
  traces and our MCP tools only.
- **03:30** The PGX stack is up:
  - Qwen3.8-27B-NVFP4 with the `qwen3_coder` tool parser
  - bge-m3 and the reranker
  - OpenSearch 3.9 with nori on loopback only

  All 9,069 chunks are indexed. Starting the Kev server on the PGX was blocked by the session's
  permission check, so it waits for the owner.
- **03:40** Hermes smoke runs against the real Qwen found four problems
  ([lesson](lessons/2026-10-23-hermes-and-qwen-serving.md)):
  - Hermes needs at least 64k context, so Qwen was restarted with 65,536.
  - `--ignore-rules` dropped the operator policy in SOUL.md, so the flag is no longer used.
  - Hermes streams its requests, so the proxy now relays SSE.
  - One proxy tag mixed up concurrent jobs, so each track now gets its own proxy.

  B0, W-llm, A1 and A2 all answer correctly on smoke questions.
- **04:00** The eval set has 410 rows: main 300, ACL 30, toolscale 80
  ([evalset.py](../rag_bench/agentic/evalset.py)).
- **04:05** At the owner's request, Kev-4B and Laya were started on the PGX loopback (ports 8009 and
  8011, reached through an ssh tunnel). One decision takes about 0.3 s on Kev and about 0.05 s on Laya.
- **04:15** Six tracks were launched on the local Qwen. Two problems:
  - Only one Hermes gateway can run per host, so A1 and A2 cannot run at the same time.
  - Generation throughput was about 17 tokens/s in total.

  The runs were stopped
  ([decision](decisions/2026-10-23-openrouter-and-subsets.md)).
- **04:30** Switched the answer LLM to OpenRouter `qwen/qwen3.8-27b` (DeepInfra pinned, thinking off).
  The owner chose plan B to stay near $2: the cheap tracks run the full set, and A1/A2 run the
  120-question `small` subset. All tracks were re-launched, as was the tool sweep.
- **05:30** The B0 and W tracks finished in about 15 minutes, and so did the tool sweep. From the
  offline cascade curve, τ was set to 0.5 for Kev and 0.7 for Laya, and both cascade tracks were run.
  Laya sent two questions down the calc path, which broke the expression JSON; that now counts as a
  wrong answer, not a crash.
- **06:20** A1 and A2 finished on the 120-question subset. Two A1 runs hit Hermes' 10-turn limit
  while searching for absent facts; these count as real results.
- **06:30** Claude Haiku 5.5 re-judged 43 answers; it agreed on correctness 43 of 43 times. Wrote the
  [results](research/2026-10-23-agentic-results.md) and
  [recommendations](decisions/2026-10-23-recommendations.md). The total spend was about $3.3.

