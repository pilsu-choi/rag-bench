"""Benchmark one store at one scale -> results/<store>_<scale>.json.

    python -m rag_bench.bench <faiss|pgvector|milvus|opensearch> [--scale base|full]

base = 1000 docs (~9k chunks); full = base + 200k distractor chunks. The DB container must be up.
Parts: build cost, ANN recall-vs-latency sweep (ground truth = exact kNN), concurrency QPS,
filtered search, and retrieval quality on the synthetic qrels (dense / BM25 / hybrid / +rerank).
"""
import argparse, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from ranx import Qrels, Run, evaluate
from rag_bench import stores
from rag_bench.common import load, pct, rerank, timed, vectors

K, EFS, EF, FILTERS, CONC = 10, [16, 32, 64, 128, 256, 512], 128, [1, 10, 50], [1, 8, 32]
DEPTH, RERANK_DEPTH, RRF_K = 100, 50, 60
CONTAINER = {"pgvector": "bench-pgvector", "milvus": "bench-milvus", "opensearch": "bench-opensearch"}


def corpus(scale):
    df = load("base")
    vec = vectors("base")
    if scale == "full":
        df, vec = pd.concat([df, load("distractors")], ignore_index=True), np.vstack([vec, vectors("distractors")])
    return df, vec


def exact_topk(q, vec, tenants, k):
    """Exact inner-product top-k ids (row indices) for each filter in FILTERS plus None (no filter)."""
    out = {}
    for f in [None, *FILTERS]:
        rows = np.arange(len(vec)) if f is None else np.flatnonzero(tenants < f)
        s = q @ vec[rows].T
        top = np.argpartition(-s, k, axis=1)[:, :k]
        out[str(f)] = rows[np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)]
    return out


def ground_truth(scale, q, vec, tenants):
    path = f"data/gt_{scale}.npz"
    if not os.path.exists(path):
        np.savez(path, **exact_topk(q, vec, tenants, K))
    return dict(np.load(path))


def recall(got, truth):
    return float(np.mean([len(set(g) & set(t)) / len(t) for g, t in zip(got, truth)]))


def run_queries(fn, args):
    res, ms = zip(*(timed(fn, *a) for a in args))
    return list(res), pct(ms)


def qps(store, q, conc, n=2000):
    with ThreadPoolExecutor(conc) as ex:
        t = time.perf_counter()
        list(ex.map(lambda i: store.search(q[i % len(q)], K), range(n)))
    return round(n / (time.perf_counter() - t), 1)


def rrf(*lists):
    s = {}
    for lst in lists:
        for r, d in enumerate(lst):
            s[d] = s.get(d, 0) + 1 / (RRF_K + r + 1)
    return sorted(s, key=s.get, reverse=True)


def reranked(qs, cands, text, cache_path="data/rerank_cache.json"):
    """Rerank top RERANK_DEPTH candidates; scores are cached on disk (identical across stores)."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = []
    for query, c in zip(qs, cands):
        c = c[:RERANK_DEPTH]
        miss = [d for d in c if f"{query}\t{d}" not in cache]
        if miss:
            cache.update({f"{query}\t{d}": s for d, s in zip(miss, rerank(query, [text[d] for d in miss]))})
        out.append(sorted(c, key=lambda d: -cache[f"{query}\t{d}"]))
    json.dump(cache, open(cache_path, "w"))
    return out


def quality(runs, qids, qrels):
    metrics = ["ndcg@10", "mrr@10", "recall@10", "recall@50", "recall@100"]
    return {name: {m: round(float(v), 4) for m, v in evaluate(qrels, Run({q: {d: 1 / (r + 1) for r, d in enumerate(lst)}
                                                                  for q, lst in zip(qids, ranked)}), metrics).items()}
            for name, ranked in runs.items()}


def mem_usage(name):
    c = CONTAINER.get(name)
    if not c:
        return None
    return subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", c],
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store", choices=stores.NAMES)
    ap.add_argument("--scale", choices=["base", "full"], default="base")
    a = ap.parse_args()
    df, vec = corpus(a.scale)
    queries, q = load("queries"), vectors("queries")
    ids, tenants = df.chunk_id.to_numpy(), df.tenant.to_numpy()
    gt = {f: ids[v] for f, v in ground_truth(a.scale, q, vec, tenants).items()}
    out = {"store": a.store, "scale": a.scale, "n_vectors": len(df), "n_queries": len(q)}

    store = stores.get(a.store, vec.shape[1])
    _, out["build_s"] = timed(store.build, ids, vec, df.text.tolist(), tenants)
    out["build_s"] = round(out["build_s"] / 1000, 1)
    out["size_bytes"], out["container_mem"] = store.size_bytes(), mem_usage(a.store)
    print(f"[{a.store}/{a.scale}] built {len(df)} in {out['build_s']}s", flush=True)

    store.set_ef(EFS[-1])
    run_queries(store.search, [(v, K) for v in q])  # warm-up: touch the whole graph once
    out["ann"] = []
    for ef in EFS:
        store.set_ef(ef)
        got, lat = run_queries(store.search, [(v, K) for v in q])
        out["ann"].append({"ef": ef, "recall@10": round(recall(got, gt["None"]), 4), **lat})
        print(f"  ef={ef} {out['ann'][-1]}", flush=True)

    store.set_ef(EF)
    out["qps"] = {c: qps(store, q, c) for c in CONC}
    out["filtered"] = []
    for f in FILTERS:
        got, lat = run_queries(store.search, [(v, K, f) for v in q])
        out["filtered"].append({"selectivity_pct": f, "recall@10": round(recall(got, gt[str(f)]), 4), **lat})
    print(f"  qps={out['qps']} filtered={out['filtered']}", flush=True)

    # Retrieval quality on synthetic qrels (relevant = source chunk of each query).
    store.set_ef(256)
    qrels = Qrels({r.qid: {r.chunk_id: 1} for r in queries.itertuples()})
    dense, out["dense_latency"] = run_queries(store.search, [(v, DEPTH) for v in q])
    runs = {"dense": dense}
    try:
        runs["bm25"], out["bm25_latency"] = run_queries(store.bm25, [(t, DEPTH) for t in queries["query"]])
        runs["hybrid"] = [rrf(d, b) for d, b in zip(dense, runs["bm25"])]
    except NotImplementedError:
        pass
    best = "hybrid" if "hybrid" in runs else "dense"
    runs[f"{best}+rerank"] = reranked(queries["query"].tolist(), runs[best], dict(zip(ids, df.text)))
    out["quality"] = quality(runs, queries.qid.tolist(), qrels)
    for name, m in out["quality"].items():
        print(f"  {name:15s} {m}", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/{a.store}_{a.scale}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
