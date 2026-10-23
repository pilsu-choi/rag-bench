"""Tool-selection sweep (PRD H4/H7): with 4, 10 and 20 tools, which tool does each decision maker pick?

    uv run --extra agentic python -m rag_bench.agentic.toolsel <llm-toolcall|llm|laya|kev> [--n 4 10 20] [--workers N]
    -> results/agentic/toolsel/<decider>.jsonl  (qid, n_tools, pred, prob, ms)

`llm-toolcall` is native Qwen tool calling: every tool is an OpenAI function and `refuse` is a function
too, so "no tool call" means no_tool. The other deciders pick a label from {tool: description}.
Only the choice is scored, not the arguments; questions come from the main and toolscale splits.
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from rag_bench.agentic import deciders, llm
from rag_bench.agentic.catalog import tool_set
from rag_bench.agentic.run import EVAL

ARG = {"search_documents": "query", "calculator": "expression", "refuse": "reason"}
SYSTEM = "당신은 사내 업무 도우미입니다. 필요하면 도구를 호출하세요. 도구가 필요 없는 대화면 바로 답하세요."


def functions(tools: dict) -> list[dict]:
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {ARG.get(name, "request"): {"type": "string"}},
                       "required": [ARG.get(name, "request")]}}}
        for name, desc in tools.items() if name != "no_tool"]


def toolcall(question, tools):
    msg = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                   tools=functions(tools), max_tokens=256)
    calls = msg.get("tool_calls") or []
    return (calls[0]["function"]["name"] if calls else "no_tool"), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decider")
    ap.add_argument("--n", type=int, nargs="+", default=[4, 10, 20])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    decide = toolcall if a.decider == "llm-toolcall" else deciders.get(a.decider).decide
    df = pd.read_parquet(EVAL)
    df = df[df.split.isin(["main", "toolscale"])]
    out = f"results/agentic/toolsel/{a.decider}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["qid"], r["n_tools"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    lock = threading.Lock()
    todo = [(r, n, tool_set(n)) for n in a.n for r in df.itertuples()
            if (r.qid, n) not in done and r.gold_tool in tool_set(n)]  # toolscale rows only once their tool exists

    def one(item):
        r, n, tools = item
        t = time.perf_counter()
        pred, p = decide(r.question, tools)
        rec = {"qid": r.qid, "n_tools": n, "gold": r.gold_tool, "pred": pred, "prob": round(float(p), 4),
               "ms": round((time.perf_counter() - t) * 1000, 1)}
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    llm.tag(track=f"toolsel-{a.decider}")
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
