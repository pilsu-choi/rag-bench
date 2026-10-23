"""Score answers -> results/agentic/<track>/<split>.judged.jsonl (resumes).

    uv run --extra agentic python -m rag_bench.agentic.judge <track> [--split main]

- refusal expected (out_of_corpus, acl): correct iff the answer refuses
- calc / doc_calc: correct iff a number in the answer equals the reference (0.5 % tolerance)
- chitchat: correct iff a non-empty reply that is not a refusal and used no retrieval
- doc_qa / multi_doc: LLM judge against the reference answer; plus faithfulness to the passages the
  track retrieved, and whether it cites a [chunk_id]
The judge is the same LLM behind the proxy (tagged "judge"); JUDGE_URL/JUDGE_MODEL/JUDGE_KEY switch it
to another endpoint for the cross-check.
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
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def refused(a: str) -> bool:
    return REFUSAL in a or "찾을 수 없" in a


def number_match(answer: str, ref: str) -> bool:
    want = float(ref.replace(",", ""))
    got = [float(x.replace(",", "")) for x in NUM.findall(answer.replace(" ", "")) if x.strip(",.")]
    return any(abs(g - want) <= max(abs(want) * 0.005, 1e-6) for g in got)


def judge_llm(prompt):
    url = os.environ.get("JUDGE_URL")
    if not url:
        return llm.ask(prompt, schema=SCHEMA, max_tokens=64)
    r = httpx.post(url, timeout=300, headers={"authorization": f"Bearer {os.environ.get('JUDGE_KEY', '')}"},
                   json={"model": os.environ["JUDGE_MODEL"], "temperature": 0, "max_tokens": 64,
                         "messages": [{"role": "user", "content": prompt}],
                         "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": SCHEMA}}})
    return json.loads(r.json()["choices"][0]["message"]["content"])


def score(row, res) -> dict:
    a = str(res.get("answer") or "")
    retrieved = res.get("retrieved") or []
    out = {"refused": refused(a), "n_retrieved": len(retrieved),
           "gold_hit": bool(set(row.gold_chunks) & set(retrieved)) if len(row.gold_chunks) else None,
           "gold_all": set(row.gold_chunks) <= set(retrieved) if len(row.gold_chunks) else None}
    if row.split == "acl" or row.intent == "out_of_corpus":
        out["correct"] = out["refused"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--split", default="main")
    a = ap.parse_args()
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
