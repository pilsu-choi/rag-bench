"""Score answers -> results/agentic/<track>/<split>.judged.jsonl (resumes).

    uv run --extra agentic python -m rag_bench.agentic.judge <track> [--split main]

- refusal expected (out_of_corpus, acl): correct iff the answer refuses (fixed phrase, or the LLM judge
  says it declined without offering an answer)
- calc / doc_calc: correct iff a number in the answer equals the reference (0.5 % tolerance)
- chitchat: correct iff a non-empty reply that is not a refusal and used no retrieval
- doc_qa / multi_doc: LLM judge against the reference answer; plus faithfulness to the passages the
  track retrieved, and whether it cites a [chunk_id]
The judge is the same LLM behind the proxy (tagged "judge"); JUDGE_URL/JUDGE_MODEL/JUDGE_KEY switch it
to another endpoint for the cross-check (`--cross 50`).
"""
import argparse, json, os, re
import httpx, pandas as pd
from rag_bench.agentic import llm
from rag_bench.agentic.catalog import REFUSAL
from rag_bench.agentic.run import EVAL

SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}, "faithful": {"type": "boolean"}},
          "required": ["correct", "faithful"]}
PROMPT = """질문에 대한 시스템 답변을 채점하세요.
- correct: 답변이 참조 답변과 같은 핵심 사실을 담고 있으면 true (표현 차이는 무시, 질문이 요구한 내용을 빠뜨리면 false)
- faithful: 답변의 모든 사실 주장이 "시스템이 본 검색 결과"로 뒷받침되면 true (검색 결과가 없으면 false)

질문: {question}
참조 답변: {ref}
시스템이 본 검색 결과:
{passages}

시스템 답변: {answer}"""
DECLINE = {"type": "object", "properties": {"declined": {"type": "boolean"}}, "required": ["declined"]}
DECLINE_PROMPT = """아래 답변이 질문에 대한 사실 답을 내놓지 않고 "알 수 없다/찾을 수 없다/답할 수 없다"는 취지로 거절했으면
declined=true, 어떤 형태로든 답(추측, 예시, 일반 지식 포함)을 제시했으면 false.

질문: {question}
답변: {answer}"""
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def refused(a: str) -> bool:
    return REFUSAL in a or "찾을 수 없" in a


def number_match(answer: str, ref: str) -> bool:
    want = float(ref.replace(",", ""))
    got = [float(x.replace(",", "")) for x in NUM.findall(answer.replace(" ", "")) if x.strip(",.")]
    return any(abs(g - want) <= max(abs(want) * 0.005, 1e-6) for g in got)


def judge_llm(prompt, schema=SCHEMA):
    url = os.environ.get("JUDGE_URL")
    if not url:
        return llm.ask(prompt, schema=schema, max_tokens=64)
    r = httpx.post(url, timeout=300, headers={"authorization": f"Bearer {os.environ.get('JUDGE_KEY', '')}"},
                   json={"model": os.environ["JUDGE_MODEL"], "temperature": 0, "max_tokens": 400,
                         "messages": [{"role": "user", "content": prompt}],
                         "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}})
    for _ in range(3):  # occasional empty content from the provider
        content = r.json()["choices"][0]["message"].get("content")
        if content:
            return json.loads(content)
        r = httpx.post(url, timeout=300, headers=r.request.headers, content=r.request.content)
    raise RuntimeError(f"judge returned no content: {r.json()['choices'][0]}")


def score(row, res) -> dict:
    a = str(res.get("answer") or "")
    retrieved = res.get("retrieved") or []
    out = {"refused": refused(a), "n_retrieved": len(retrieved),
           "gold_hit": bool(set(row.gold_chunks) & set(retrieved)) if len(row.gold_chunks) else None,
           "gold_all": set(row.gold_chunks) <= set(retrieved) if len(row.gold_chunks) else None}
    if row.split == "acl" or row.intent == "out_of_corpus":  # agents refuse in their own words
        out["correct"] = out["refused"] or judge_llm(DECLINE_PROMPT.format(question=row.question, answer=a),
                                                     DECLINE)["declined"]
    elif row.intent in ("calc", "doc_calc"):
        out["correct"] = number_match(a, row.ref_answer)
    elif row.intent == "chitchat":
        out["correct"] = bool(a.strip()) and not out["refused"] and not retrieved
    else:
        j = judge_llm(PROMPT.format(question=row.question, ref=row.ref_answer, answer=a,
                                    passages=res.get("passages") or "(없음)"))
        out |= {"correct": j["correct"] and not out["refused"], "faithful": j["faithful"],
                "cited": bool(re.search(r"\[[^\]]+#\d+\]", a))}
    return out


def crosscheck(n: int):
    """Re-judge n LLM-judged answers (sampled across tracks) with the JUDGE_* model -> agreement with Qwen."""
    ev = pd.read_parquet(EVAL).set_index("qid")
    rows = []
    for track in sorted(os.listdir("results/agentic")):
        f = f"results/agentic/{track}/main.judged.jsonl"
        if os.path.exists(f):
            res = {r["qid"]: r for r in map(json.loads, open(f"results/agentic/{track}/main.jsonl"))}
            rows += [(track, j, res[j["qid"]]) for j in map(json.loads, open(f)) if "faithful" in j]
    out = []
    for i in pd.Series(range(len(rows))).sample(n, random_state=0):
        track, j, res = rows[i]
        row = ev.loc[j["qid"]]
        try:
            k = judge_llm(PROMPT.format(question=row.question, ref=row.ref_answer, answer=res.get("answer") or "",
                                        passages=res.get("passages") or "(없음)"))
        except (RuntimeError, ValueError) as e:  # counted as skipped, not as disagreement
            print(track, j["qid"], str(e)[:200])
            continue
        out.append({"track": track, "qid": j["qid"], "qwen": j["correct"],
                    "other": k["correct"] and not j["refused"], "qwen_faithful": j["faithful"], "other_faithful": k["faithful"]})
    with open("results/agentic/judge_crosscheck.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    d = pd.DataFrame(out)
    print(f"correct agreement {(d.qwen == d.other).mean():.3f}, faithful agreement "
          f"{(d.qwen_faithful == d.other_faithful).mean():.3f}, qwen acc {d.qwen.mean():.3f} vs other {d.other.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", nargs="?")
    ap.add_argument("--split", default="main")
    ap.add_argument("--cross", type=int, help="cross-check N judgments with JUDGE_URL/JUDGE_MODEL/JUDGE_KEY")
    a = ap.parse_args()
    if a.cross:
        return crosscheck(a.cross)
    ev = pd.read_parquet(EVAL).set_index("qid")
    src, dst = f"results/agentic/{a.track}/{a.split}.jsonl", f"results/agentic/{a.track}/{a.split}.judged.jsonl"
    done = {(r["qid"], r["rep"]) for r in map(json.loads, open(dst))} if os.path.exists(dst) else set()
    llm.tag(track="judge", of=a.track, split=a.split)
    for res in map(json.loads, open(src)):
        if (res["qid"], res["rep"]) in done:
            continue
        row = ev.loc[res["qid"]]
        rec = {"qid": res["qid"], "rep": res["rep"], "intent": row.intent, **score(row, res)}
        with open(dst, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
