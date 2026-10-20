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
