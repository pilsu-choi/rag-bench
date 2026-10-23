"""Build results/agentic/REPORT.md from the judged JSONL, the proxy log and the tool-selection sweep.

    uv run --extra agentic python -m rag_bench.agentic.report
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from rag_bench.agentic.run import EVAL

R = "results/agentic"
TRACKS = ["b0", "w-llm", "w-laya", "w-kev", "w-cascade-laya-0.7", "w-cascade-kev-0.5", "a1", "a2"]


def jl(path):
    return pd.DataFrame(map(json.loads, open(path))) if os.path.exists(path) else pd.DataFrame()


def md(df, floatfmt=".3f"):
    return df.to_markdown(floatfmt=floatfmt) + "\n" if len(df) else "(no data)\n"


def attribute(calls, ev):
    """Calls logged without a qid (runs with --workers) get the qid whose question text they contain."""
    qs = list(ev.question.items())
    text = calls.messages.apply(lambda m: " ".join(str(x.get("content")) for x in m if x.get("role") == "user"))
    found = text.apply(lambda t: next((q for q, s in qs if s in t), None))
    calls["qid"] = calls.get("qid", pd.Series(index=calls.index)).fillna(found)
    calls["rep"] = calls.get("rep", pd.Series(index=calls.index)).fillna(-1)
    return calls


def llm_cost(calls, track, split):
    c = calls[(calls.track == track) & (calls.split == split)] if len(calls) else calls
    if not len(c):
        return {}
    c = c.assign(rep=0) if split != "consistency" else c  # unattributed reps fold together
    per_q = c.groupby(["qid", "rep"]).agg(calls=("ms", "size"), tok_in=("prompt_tokens", "sum"),
                                         tok_out=("completion_tokens", "sum"), llm_ms=("ms", "sum"))
    return per_q.mean().round(1).to_dict()


def main_table(ev, calls, qids=None):
    rows, by_intent = {}, {}
    for t in TRACKS:
        res, jd = jl(f"{R}/{t}/main.jsonl"), jl(f"{R}/{t}/main.judged.jsonl")
        if not len(jd):
            continue
        if qids is not None:
            jd = jd[jd.qid.isin(qids)]
        m = jd.merge(res[["qid", "rep", "ms"] + (["intent"] if "intent" in res else [])].rename(
            columns={"intent": "pred_intent"}), on=["qid", "rep"])
        doc = m[m.intent.isin(["doc_qa", "multi_doc", "doc_calc"])]
        rows[t] = {"accuracy": m.correct.mean(), "p50 ms": m.ms.median(), "p95 ms": m.ms.quantile(.95),
                   "gold hit (doc)": doc.gold_hit.mean(), "faithful": m.get("faithful", pd.Series()).mean(),
                   "false refusal (doc)": doc.refused.mean(),
                   "refusal (ooc)": m[m.intent == "out_of_corpus"].refused.mean(),
                   **llm_cost(calls, t, "main")}
        if "pred_intent" in m:
            rows[t]["intent macro-F1"] = f1_score(m.intent, m.pred_intent.fillna("none"), average="macro")
        by_intent[t] = m.groupby("intent").correct.mean()
    return pd.DataFrame(rows).T, pd.DataFrame(by_intent)


def decisions(track):
    """Per-question intent decision (label, prob, ms) from a W-track run."""
    res = jl(f"{R}/{track}/main.jsonl")
    if not len(res):
        return pd.DataFrame()
    d = res.apply(lambda r: next((s for s in (r.steps if isinstance(r.steps, list) else []) if s["step"] == "intent"), {}), axis=1).apply(pd.Series)
    return pd.concat([res[["qid"]], d.reindex(columns=["label", "prob", "ms"])], axis=1).dropna().set_index("qid")


def cascade_curve(ev, model):
    """Offline cascade on the intent decision: model if prob >= tau, else the LLM's label."""
    dm, dl = decisions(f"w-{model}"), decisions("w-llm")
    if not len(dm) or not len(dl):
        return pd.DataFrame()
    j = dm.join(dl, rsuffix="_llm", how="inner").join(ev.intent)
    out = []
    for tau in [0, .5, .6, .7, .8, .9, .95, 1.01]:
        use = j.prob >= tau
        pred = np.where(use, j.label, j.label_llm)
        out.append({"tau": tau, "handled by model": use.mean(), "intent acc": (pred == j.intent).mean(),
                    "mean decision ms": (j.ms + (~use) * j.ms_llm).mean()})
    return pd.DataFrame(out).set_index("tau")


def ece(ev, model, bins=10):
    d = decisions(f"w-{model}").join(ev.intent)
    if not len(d):
        return None
    ok, p = (d.label == d.intent).astype(float), d.prob
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(ok[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def side_table(split, metric):
    rows = {}
    for t in TRACKS:
        jd = jl(f"{R}/{t}/{split}.judged.jsonl")
        if len(jd):
            rows[t] = metric(jd)
    return pd.DataFrame(rows).T


def consistency(jd):
    g = jd.groupby("qid")
    return {"answer-correct agreement": g.correct.nunique().eq(1).mean(), "questions": g.ngroups}


def toolsel():
    rows = {}
    for f in sorted(glob.glob(f"{R}/toolsel/*.jsonl")):
        d = jl(f)
        name = os.path.basename(f)[:-6]
        for n, g in d.groupby("n_tools"):
            lenient = g.gold.replace("refuse", "no_tool") == g.pred.replace("refuse", "no_tool")
            rows[(name, n)] = {"accuracy": (g.gold == g.pred).mean(), "refuse=no_tool": lenient.mean(),
                               "p50 ms": g.ms.median(), "questions": len(g)}
    return pd.DataFrame(rows).T


def main():
    ev = pd.read_parquet(EVAL).set_index("qid")
    calls = jl(f"{R}/llm_calls.jsonl")
    if len(calls):
        u = calls.pop("usage").apply(lambda x: x or {}).apply(pd.Series)
        calls = calls.join(u[["prompt_tokens", "completion_tokens"]])
        calls["split"] = calls.get("split", pd.Series(index=calls.index)).fillna("main")
        calls = attribute(calls, ev)
    overall, by_intent = main_table(ev, calls)
    small, small_intent = main_table(ev, calls, set(ev.index[ev.small & (ev.split == "main")]))
    parts = ["# Part 2 results\n", "Generated by `rag_bench.agentic.report`. Metrics are defined in PRD.md.\n",
             "## Main set (300; A1/A2 ran only the 120-question subset, so their row is that subset)\n", md(overall),
             "## Accuracy by intent\n", md(by_intent),
             "## Same 120-question subset for every track (architecture comparison)\n", md(small), md(small_intent),
             "## ACL set (answer only from another tenant; correct = refusal)\n",
             "The filter is injected by the tool server, so the gold chunk should never be retrieved. Other chunks of\n"
             "the same article can carry tenants < 50, so an answer is not necessarily a leak.\n\n",
             md(side_table("acl", lambda d: {"gold chunk retrieved": d.gold_hit.mean(), "answered": 1 - d.correct.mean(),
                                             "questions": len(d)})),
             "## Consistency (3 runs)\n", md(side_table("consistency", consistency)),
             "## Tool selection by number of tools\n", md(toolsel())]
    for m in ["laya", "kev"]:
        parts += [f"## Cascade on intent: {m} → LLM (offline, from the W-{m} and W-llm runs)\n",
                  f"ECE of the intent decision: {ece(ev, m):.3f}. τ for the W-cascade track was picked from this "
                  "same curve, so its gain is optimistic.\n\n", md(cascade_curve(ev, m))]
    open(f"{R}/REPORT.md", "w").write("\n".join(parts))
    print(open(f"{R}/REPORT.md").read())


if __name__ == "__main__":
    main()
