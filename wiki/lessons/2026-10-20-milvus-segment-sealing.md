---
type: Lesson
title: Milvus seals small segments by default
description: Default sealing split 209k vectors into 10 segments, which compaction did not merge; QPS stayed flat with more clients until the seal size was raised.
tags: [lesson, milvus]
timestamp: 2026-10-20T09:15:00Z
resource: ../../docker/milvus-user.yaml
---

# Milvus seals small segments by default

**What happened.** Milvus 2.6.25 with 209k vectors (1024-d) reached 210 / 224 / 351 QPS with
1 / 8 / 32 clients. Its recall at ef=16 was 0.94, while the single-graph stores got 0.79.
`get_query_segment_info` showed **10 segments of about 20k rows each**, so every query searched
10 HNSW graphs and used up the 4 CPUs. `compact()`, both normal and `is_major=True`, left all
10 segments in place.

**Cause.** `dataCoord.segment.sealProportion` defaults to 0.12 of `maxSize` (1024 MB). A growing
segment seals at about 120 MB, which is about 20k rows of 1024-d float vectors plus text.

**Fix.** `docker/milvus-user.yaml`, mounted as `/milvus/configs/user.yaml`, sets `maxSize: 2048`
and `sealProportion: 0.9`. That gives 2 segments (75k and 134k rows) and 352 / 777 / 589 QPS. The
recall-vs-latency curve then lines up with OpenSearch's.

**Takeaway.**
- Segment layout matters as much as HNSW parameters for Milvus throughput.
- Check segment count after loading.
- Comparing an untuned Milvus against a force-merged OpenSearch would not be fair.
