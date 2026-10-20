---
type: Decision
title: Benchmark design
description: Separate the ANN engine from the retrieval pipeline, use identical vectors, compare at equal recall, and run one container at a time.
tags: [decision, methodology]
timestamp: 2026-10-20T09:15:00Z
resource: ../../rag_bench/bench.py
---

# Benchmark design

## Context

Search accuracy in RAG mostly comes from the embedding model, chunking, hybrid search and
reranking. The vector store mainly affects ANN approximation loss, latency, filtering and what
search features it offers. Measuring both at once would mix up the two effects.

## Decision

1. **Two parts.** (a) The ANN engine is scored by recall@10 against **exact kNN**, so the embedding
   model drops out. Latency and QPS are measured too. (b) The retrieval pipeline is scored by
   nDCG@10, MRR@10 and Recall@10/50/100 on synthetic qrels, for dense, BM25, hybrid and +rerank.
2. **Same vectors everywhere.** Embeddings are computed once and stored as `.npy`. Model latency
   is measured separately in `model_latency.py`.
3. **Same HNSW settings** (M=16, efConstruction=200). `ef_search` is swept from 16 to 512, and the
   report compares stores **at the smallest ef that reaches recall ≥ 0.95**, because the same ef means
   different things in different engines.
4. **Client-side RRF** (k=60) for hybrid in every store, so that the fusion is identical. Native
   hybrid APIs are not compared.
5. **One container at a time**, each limited to 4 CPU and 6 GB. FAISS runs in-process with 4 OpenMP
   threads.
6. **Scale.** 1,000 documents (~9k chunks) is enough for quality, but every store answers within a few ms
   at that size. 200k distractor chunks are added to make the latency differences visible.
7. **Filters.** Each chunk gets a random `tenant` from 0 to 99, and the filter `tenant < x` gives
   1, 10 and 50 % selectivity.
8. **Fair index layout.** OpenSearch is force-merged to one segment. Milvus is configured to seal
   large segments ([lesson](../lessons/2026-10-20-milvus-segment-sealing.md)).

## Alternatives considered

- VectorDBBench is quick for part (a), but it is made by the Milvus vendor and cannot run Korean
  BM25, hybrid search or reranking against our own qrels.
- Native hybrid search (OpenSearch, Milvus) could be added later as an extra pipeline per store.
