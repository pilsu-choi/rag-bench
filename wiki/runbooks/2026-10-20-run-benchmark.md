---
type: Runbook
title: Run the benchmark end to end
description: Serve the models on the PGX, build the corpus and queries, embed once, benchmark every store, write the report.
tags: [runbook, vllm, docker]
timestamp: 2026-10-20T09:15:00Z
resource: ../../scripts/run-all.sh
---

# Run the benchmark end to end

## 1. Models on the PGX

```bash
# on the PGX, weights under /mnt/ssd/models (download as the user, not inside a container)
uvx --from 'huggingface_hub[hf_xet]' hf download BAAI/bge-m3 pytorch_model.bin --local-dir bge-m3
uvx --from 'huggingface_hub[hf_xet]' hf download BAAI/bge-m3 --local-dir bge-m3 --exclude 'onnx/*' --exclude '*.bin'
uvx --from 'huggingface_hub[hf_xet]' hf download BAAI/bge-reranker-v2-m3 --local-dir bge-reranker-v2-m3 --exclude '*.bin'
bash serve-retrieval.sh       # containers "embed" (:8001) and "rerank" (:8002), ~1 min to ready
```

For query synthesis, an OpenAI-compatible LLM must be up on `:8000` (Qwen3.8-27B, see the
dgx-spark-llm-serving repo). Stop it afterwards (`docker rm -f qwen`) to free GPU memory.

## 2. Data, queries, embeddings (workstation)

```bash
uv sync
export PGX_HOST=<model host>
uv run python -m rag_bench.prep            # ~1 min, downloads one 205 MB Wikipedia shard
uv run python -m rag_bench.synth --n 600   # ~3.5 min at 8 concurrent requests
uv run python -m rag_bench.embed_all       # ~15 min, 209k chunks at ~300 texts/s
```

## 3. Benchmark and report

```bash
scripts/run-all.sh                 # ~50 min; STORES="milvus opensearch" to run a subset
uv run python -m rag_bench.report  # results/REPORT.md, results/ann_pareto.png
```

`run-all.sh` starts one DB with `docker compose -f docker/<store>.yml up -d`, waits for it to be ready,
runs `bench.py` at the base and full scales, and then tears it down with `down -v`.

## Checks

- `curl $PGX_HOST:8001/v1/models` and `:8002` respond.
- `data/gt_*.npz` (exact-kNN ground truth) and `data/rerank_cache.json` are caches. Delete them after
  changing the corpus or the queries.
