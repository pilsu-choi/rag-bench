"""Per-query model latency on the PGX (store-independent part of the pipeline) -> results/models.json."""
import json
from rag_bench.common import embed, load, pct, rerank, timed

if __name__ == "__main__":
    qs, texts = load("queries")["query"].tolist()[:100], load("base").text.tolist()
    _, emb = zip(*(timed(embed, [q]) for q in qs))
    _, rr = zip(*(timed(rerank, q, texts[i * 50 : (i + 1) * 50]) for i, q in enumerate(qs)))
    out = {"embed_1_query": pct(emb), "rerank_50_docs": pct(rr)}
    json.dump(out, open("results/models.json", "w"), indent=1)
    print(out)
