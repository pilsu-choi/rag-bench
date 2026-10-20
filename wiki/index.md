---
type: Index
title: rag-bench wiki
description: "Entry point: benchmarking FAISS, pgvector, Milvus and OpenSearch as the retriever of a Korean RAG app."
tags: [index]
timestamp: 2026-10-20T09:15:00Z
---

# rag-bench wiki

This wiki follows the Open Knowledge Format (OKF v0.1) used in my other projects. Pages are plain
Markdown files with YAML frontmatter, one concept per file, linked with ordinary relative links. New
files have a `YYYY-MM-DD-` prefix. Each directory has an `index.md`, and [log.md](log.md) holds the
chronological history of the work.

Start with the [overview](overview.md).

| Section | What is there |
| --- | --- |
| [overview.md](overview.md) | Goal, setup, current status |
| [runbooks/](runbooks/index.md) | Step-by-step: reproducing the benchmark |
| [decisions/](decisions/index.md) | Benchmark design and model choices, and why |
| [lessons/](lessons/index.md) | Things that went wrong and what they taught |
| [research/](research/index.md) | Measured results and what they mean |
| [issues/](issues/index.md) | Known gaps and open questions |
| [log.md](log.md) | Chronological work log |

## Conventions

- Every page has `type`, `title`, `description`, `tags` and `timestamp` (UTC, ISO 8601), plus
  `resource` on pages about code.
- Numbers come from the 2026-10-20 run. The raw per-store JSON is in [results/](../results/).
- No secrets, IP addresses or network names are recorded here.
