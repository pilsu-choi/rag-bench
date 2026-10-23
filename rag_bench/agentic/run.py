"""Run one track over the eval set -> results/agentic/<track>/<split>.jsonl (resumes; one line per question).

    uv run --extra agentic python -m rag_bench.agentic.run <track> [--split main|acl|consistency] [--reps 3]

Tracks: b0 | w-llm | w-laya | w-kev | w-cascade-<laya|kev>-<tau> | a1 | a2.
Sequential (concurrency 1) so latencies are comparable; the logging proxy must be running.
The acl split runs with TENANT_LT=50 (the user may read only tenants 0-49).
"""
import argparse, json, os, time
import pandas as pd
from rag_bench.agentic import deciders, llm, workflow

EVAL = "data/agentic_eval.parquet"


def solver(track: str):
    if track == "b0":
        return lambda q: workflow.run(q)
    if track.startswith("w-cascade-"):
        _, _, model, tau = track.split("-")
        d = deciders.Cascade(deciders.get(model), deciders.get("llm"), float(tau))
        return lambda q: workflow.run(q, d)
    if track.startswith("w-"):
        d = deciders.get(track[2:])
        return lambda q: workflow.run(q, d)
    if track in ("a1", "a2"):
        from rag_bench.agentic import hermes
        return lambda q: hermes.run(q, track)
    raise SystemExit(f"unknown track {track}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--split", default="main")
    ap.add_argument("--reps", type=int, default=3, help="repetitions for the consistency split")
    a = ap.parse_args()
    df = pd.read_parquet(EVAL)
    rows, reps = (df[df.consistency], a.reps) if a.split == "consistency" else (df[df.split == a.split], 1)
    if a.split == "acl":
        os.environ["TENANT_LT"] = "50"
    out = f"results/agentic/{a.track}/{a.split}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["qid"], r["rep"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    solve = solver(a.track)
    for rep in range(reps):
        for r in rows.itertuples():
            if (r.qid, rep) in done:
                continue
            llm.tag(track=a.track, split=a.split, qid=r.qid, rep=rep)
            t = time.perf_counter()
            try:
                res = solve(r.question)
            except Exception as e:  # recorded, scored as a failure
                res = {"answer": "", "error": repr(e)}
            res |= {"qid": r.qid, "rep": rep, "ms": round((time.perf_counter() - t) * 1000, 1)}
            with open(out, "a") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"{a.track} {r.qid} {res['ms']:.0f} ms  {str(res.get('answer'))[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
