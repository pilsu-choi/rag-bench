---
type: Decision
title: Model choice
description: bge-m3 for embeddings and bge-reranker-v2-m3 for reranking, served by vLLM on the PGX; Qwen3.8-27B only for query synthesis.
tags: [decision, models, vllm]
timestamp: 2026-10-20T09:15:00Z
resource: ../../scripts/serve-retrieval.sh
---

# Model choice

- **Embedding:** `BAAI/bge-m3` (1024-d, multilingual, strong on Korean). It ships only
  `pytorch_model.bin`, which vLLM loads fine.
- **Reranker:** `BAAI/bge-reranker-v2-m3`. Its scores are sigmoid scores, about 0.98 for a relevant
  passage and about 1e-5 for an unrelated one, which was checked by hand before use.
- **Serving:** the `vllm/vllm-openai:v0.31.0` image (multi-arch, arm64), already on the PGX.
  `--runner pooling` exposes `/v1/embeddings` and `/rerank`. Each container gets
  `--gpu-memory-utilization 0.15` so that both fit side by side.
- **Query synthesis:** Qwen3.8-27B NVFP4 with thinking off, generating one paraphrased question per
  chunk. It was stopped afterwards.
- **Measured on the PGX:** one query embeds in 16 ms (p50); 50 passages rerank in 240 ms (p50).
  Throughput is ~300 chunks/s for 500-character chunks at 8 concurrent batches of 128.

Not tried yet: Qwen3-Embedding or Qwen3-Reranker, or bge-m3 sparse vectors in place of BM25.
