---
type: Overview
title: Overview
description: Why rag-bench exists, what it measures, on which hardware, and where it stands.
tags: [overview, rag, vector-db]
timestamp: 2026-10-23T02:35:00Z
resource: ../README.md
---

# Overview

## Goal

The goal is to pick a vector store for a Korean RAG application, judged on **retrieval only**:
latency, throughput and search accuracy, without the generation step. The candidates are FAISS,
PostgreSQL with pgvector (ParadeDB image), Milvus and OpenSearch.

## Setup

- **Models (DGX Spark-class PGX, GB10, vLLM 0.31.0):** `BAAI/bge-m3` embeddings and
  `BAAI/bge-reranker-v2-m3`. Qwen3.8-27B was used once to synthesise the queries and then stopped.
  See [model choice](decisions/2026-10-20-model-choice.md).
- **DBs (WSL2 workstation, 16 cores, 15 GB RAM):** one Docker container at a time, each limited to
  4 CPU and 6 GB.
- **Data:** 1,000 Korean Wikipedia articles split into 9,069 chunks (`base`). The `full` scale adds
  200,000 distractor chunks. There are 598 synthetic queries, each with one relevant chunk.

The design and the reasons for it are in [benchmark design](decisions/2026-10-20-benchmark-design.md).

## Status (2026-10-20)

- All four stores were measured at both scales, and the results are in
  [research/2026-10-20-results.md](research/2026-10-20-results.md).
- In short, the store barely changes search quality. Embedding, hybrid search and the reranker
  decide it. The stores differ on filtered search, concurrency and memory.
- Open items are listed in [issues](issues/index.md).

## Part 2: agentic RAG (started 2026-10-23)

- **Architecture question:** a general agent runtime (Hermes Agent) with retrieval tools, against a
  RAG-specific workflow, against an agent runtime with the workflow as one RAG-layer tool.
- **Tool-calling question:** whether Laya or Kev can replace LLM tool calling for tool selection, on
  its own or as a cascade in front of the LLM, and whether that is worth shipping.
- The spec is [PRD.md](../PRD.md) and the reasoning is in the
  [Part 2 design](decisions/2026-10-23-agentic-rag-design.md).
- **Status:** M0 is in progress (PGX serving stack, plus feasibility checks of Hermes, Laya and Kev).

