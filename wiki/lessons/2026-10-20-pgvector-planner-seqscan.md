---
type: Lesson
title: pgvector planner skips the HNSW index
description: On a small table the planner chose an exact sort over the HNSW index for some ef_search values; force the index for ANN benchmarks.
tags: [lesson, pgvector, postgres]
timestamp: 2026-10-20T09:15:00Z
resource: ../../rag_bench/stores/pgvector.py
---

# pgvector planner skips the HNSW index

**What happened.** The table had 9k rows. With `hnsw.ef_search` set to 32 or 64, queries took
23 ms at recall 1.0. With ef 16 or 128 they took 1–3 ms. `EXPLAIN` showed `Sort` over a full scan
at ef 32 and `Index Scan using docs_emb` at 16 and 128. The planner's cost estimate for the HNSW
scan depends on `ef_search`, and for small tables a full sort can look cheaper.

**Fix.** `search()` runs `SET LOCAL enable_seqscan = off` in the same transaction as
`hnsw.ef_search` and `hnsw.iterative_scan = relaxed_order`.

**Takeaway.**
- Check the plan before trusting pgvector latency numbers.
- In production the planner may also pick an exact sort on purpose. With a selective filter, it
  used the `tenant` B-tree and an exact sort, which gave recall 1.0. That is a real advantage of
  Postgres, so it was left in place.
