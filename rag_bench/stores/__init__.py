"""Vector store adapters. Each module defines `Store` with this interface:

    Store(dim)                                   connect (container already running)
    .build(ids, vecs, texts, tenants, m, ef_construction)
                                                 drop + create collection, insert, build HNSW
                                                 (inner product on L2-normalised vectors) and full-text
                                                 index on `texts`; return only when fully searchable
    .set_ef(ef)                                  HNSW ef_search for subsequent searches
    .search(vec, k, tenant_lt=None) -> ids       dense ANN; optional filter `tenant < tenant_lt`
    .bm25(query, k) -> ids                       Korean BM25 full-text search (NotImplementedError if none)
    .size_bytes() -> int                         index + data size on disk/memory, best effort

search/bm25 must be thread-safe (the benchmark calls them from a thread pool).
"""
import importlib

NAMES = ["faiss", "pgvector", "milvus", "opensearch"]


def get(name: str, dim: int):
    return importlib.import_module(f"rag_bench.stores.{name}").Store(dim)
