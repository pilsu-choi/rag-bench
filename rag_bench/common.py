"""Shared paths, model endpoints and timing helpers."""
import os, time
import httpx, numpy as np, pandas as pd

PGX = os.environ.get("PGX_HOST", "pgx")  # host serving the models; set PGX_HOST to its address
LLM_URL = os.environ.get("LLM_URL", f"http://{PGX}:8000/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")  # served name, or an OpenRouter model id
EMBED_URL = f"http://{PGX}:8001/v1/embeddings"
RERANK_URL = f"http://{PGX}:8002/rerank"
DATA = "data"
_http = httpx.Client(timeout=300)


def load(name: str) -> pd.DataFrame:
    """name: base | distractors | queries"""
    return pd.read_parquet(f"{DATA}/{name}.parquet")


def vectors(name: str) -> np.ndarray:
    return np.load(f"{DATA}/emb_{name}.npy")


def embed(texts: list[str]) -> np.ndarray:
    r = _http.post(EMBED_URL, json={"model": "bge-m3", "input": texts, "truncate_prompt_tokens": 8192}).json()
    if "data" not in r:
        raise RuntimeError(r)
    v = np.array([d["embedding"] for d in r["data"]], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rerank(query: str, docs: list[str]) -> list[float]:
    r = _http.post(RERANK_URL, json={"model": "bge-reranker-v2-m3", "query": query, "documents": docs}).json()
    scores = [0.0] * len(docs)
    for x in r["results"]:
        scores[x["index"]] = x["relevance_score"]
    return scores


def timed(fn, *args, **kw):
    """Return (result, elapsed_ms)."""
    t = time.perf_counter()
    out = fn(*args, **kw)
    return out, (time.perf_counter() - t) * 1000


def pct(ms: list[float]) -> dict:
    p = np.percentile(ms, [50, 95, 99])
    return {k: round(float(v), 2) for k, v in zip(("p50", "p95", "p99"), p)}
