---
type: Index
title: Issues
description: Known gaps in the 2026-10-20 run and open questions.
tags: [index, issues]
timestamp: 2026-10-20T09:15:00Z
---

# Issues

| # | Issue | Impact | Next step |
| --- | --- | --- | --- |
| 1 | OpenSearch filtered recall is ~0.54 at 1 % selectivity | Its filter numbers look worse than they need to | Set `index.knn.advanced.filtered_exact_search_threshold` and re-run |
| 2 | Warm-up was 100 queries, so the lowest-ef points of Milvus and OpenSearch are slower than the next ef | Only the low-recall part of the chart; the equal-recall comparison is unaffected | `bench.py` is already fixed; re-run `run-all.sh` |
| 3 | One relevant chunk per synthetic query | Absolute nDCG and recall are pessimistic | Judge top-k with an LLM, or use real user queries |
| 4 | Milvus `size_bytes` is 0 (no client API) | No disk-size column for Milvus | Read MinIO bucket size, or rely on container memory |
| 5 | pgvector top-100 and BM25 slow at 209k (p95 67 / 162 ms) | Real, but untuned (`shared_buffers` 1.5 GB, `halfvec` unused) | Try `halfvec`, more memory, pgvectorscale |
| 6 | Single run, workstation client on WSL2 | No error bars | Repeat 3× and report the spread |
| 7 | The owner's real documents have not been run | Results are for Wikipedia text | Swap the input in `prep.py` |
