"""Synthesize Korean queries (one per sampled chunk) with the LLM served on the PGX.

The source chunk is the single relevant passage (qrels). Usage: python -m rag_bench.synth [--n 500]
"""
import argparse, asyncio, os
import httpx, pandas as pd

LLM_URL = os.environ.get("LLM_URL", "http://172.30.1.84:8000/v1/chat/completions")
PROMPT = """다음은 '{title}' 문서의 한 단락입니다.

{text}

이 단락만 읽으면 답할 수 있는 질문 하나를 한국어로 만드세요.
- 실제 사용자가 검색창에 입력할 법한 자연스러운 질문
- 단락의 문장을 그대로 베끼지 말고 다른 표현으로 바꿔 쓸 것
- "이 문서", "위 단락" 같은 표현 없이 질문만으로 대상이 분명할 것
질문 한 줄만 출력하세요."""
BANNED = ("문서", "단락", "본문", "위 글")


async def ask(client, sem, row) -> str:
    async with sem:
        r = await client.post(LLM_URL, json={
            "model": "qwen3.8-27b", "max_tokens": 128, "temperature": 0.7,
            "messages": [{"role": "user", "content": PROMPT.format(title=row.title, text=row.text)}],
            "chat_template_kwargs": {"enable_thinking": False},
        })
        return r.json()["choices"][0]["message"]["content"].strip().splitlines()[0].strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    base = pd.read_parquet("data/base.parquet")
    src = base[base.text.str.len() >= 200].sample(a.n, random_state=1)
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=300) as c:
        qs = await asyncio.gather(*(ask(c, sem, r) for r in src.itertuples()))
    df = src[["chunk_id", "doc_id"]].assign(query=qs).reset_index(drop=True)
    df = df[(df["query"].str.len() >= 8) & ~df["query"].str.contains("|".join(BANNED))]
    df.insert(0, "qid", [f"q{i}" for i in range(len(df))])
    df.to_parquet("data/queries.parquet")
    print(f"{len(df)}/{a.n} queries kept")


if __name__ == "__main__":
    asyncio.run(main())
