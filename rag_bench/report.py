"""Aggregate results/*.json -> results/REPORT.md + results/ann_pareto.png."""
import glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rag_bench.stores import NAMES

COLORS = dict(zip(NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # fixed per store
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SCALES = ["base", "full"]


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def at_recall(ann, target=0.95):
    """Smallest-ef point reaching the target recall (fair latency comparison)."""
    return next((p for p in ann if p["recall@10"] >= target), ann[-1])


def plot(res):
    fig, axes = plt.subplots(1, len(SCALES), figsize=(11, 4.2), facecolor=SURFACE)
    for ax, scale in zip(axes, SCALES):
        ax.set_facecolor(SURFACE)
        for r in (r for r in res if r["scale"] == scale):
            xs, ys = [p["p95"] for p in r["ann"]], [p["recall@10"] for p in r["ann"]]
            ax.plot(xs, ys, color=COLORS[r["store"]], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=r["store"])
        n = next((r["n_vectors"] for r in res if r["scale"] == scale), 0)
        ax.set(xscale="log", title=f"{scale} ({n:,} vectors)", xlabel="p95 latency per query (ms, log)", ylabel="Recall@10 vs exact kNN")
        ax.title.set_color(INK)
        ax.grid(color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.suptitle("HNSW recall vs latency (M=16, efC=200; ef_search 16→512)", color=INK)
    fig.tight_layout()
    fig.savefig("results/ann_pareto.png", dpi=130)


def main():
    res = [json.load(open(p)) for p in sorted(glob.glob("results/*_*.json"))]
    res.sort(key=lambda r: (SCALES.index(r["scale"]), NAMES.index(r["store"])))
    md = ["# Results", "", "![ANN pareto](ann_pareto.png)", ""]
    md += ["## Build and ANN (k=10)", "", table(
        ["scale", "store", "vectors", "build s", "size MB", "container mem", "ef @R≥0.95", "recall", "p50 ms", "p95 ms", "QPS c=1/8/32"],
        [[r["scale"], r["store"], r["n_vectors"], r["build_s"], round(r["size_bytes"] / 2**20) if r["size_bytes"] else "-", r["container_mem"] or "-",
          (p := at_recall(r["ann"]))["ef"], p["recall@10"], p["p50"], p["p95"], " / ".join(str(round(v)) for v in r["qps"].values())] for r in res])]
    md += ["", "## Filtered search (ef=128, filter `tenant < x`, recall vs exact filtered kNN)", "", table(
        ["scale", "store"] + [f"{f['selectivity_pct']}%: recall / p95" for f in res[0]["filtered"]],
        [[r["scale"], r["store"]] + [f"{f['recall@10']} / {f['p95']}" for f in r["filtered"]] for r in res])]
    md += ["", "## Retrieval quality on synthetic qrels (598 queries)", "", table(
        ["scale", "store", "pipeline", "nDCG@10", "MRR@10", "R@10", "R@50", "R@100"],
        [[r["scale"], r["store"], name, m["ndcg@10"], m["mrr@10"], m["recall@10"], m["recall@50"], m["recall@100"]]
         for r in res for name, m in r["quality"].items()])]
    md += ["", "## Query latency, dense top-100 (ef=256) and BM25 top-100", "", table(
        ["scale", "store", "dense p50 / p95", "bm25 p50 / p95"],
        [[r["scale"], r["store"], f"{r['dense_latency']['p50']} / {r['dense_latency']['p95']}",
          f"{r['bm25_latency']['p50']} / {r['bm25_latency']['p95']}" if "bm25_latency" in r else "-"] for r in res])]
    try:
        m = json.load(open("results/models.json"))
        md += ["", "## Model latency on the PGX (ms)", "", table(["step", "p50", "p95", "p99"], [[k, *v.values()] for k, v in m.items()])]
    except FileNotFoundError:
        pass
    open("results/REPORT.md", "w").write("\n".join(md) + "\n")
    plot(res)


if __name__ == "__main__":
    main()
