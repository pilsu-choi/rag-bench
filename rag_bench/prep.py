"""Build the corpus: 1000 Korean Wikipedia articles (base) + distractor chunks for scale tests.

Usage: python -m rag_bench.prep [--distractors 200000]
"""
import argparse, re
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

CHUNK_CHARS = 500
SHARD = "20231101.ko/train-00001-of-00003.parquet"


def chunk(text: str) -> list[str]:
    """Pack paragraphs into ~CHUNK_CHARS chunks; long paragraphs are split by sentence."""
    sents = [s for p in re.split(r"\n+", text) for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s]
    units = [s[i : i + CHUNK_CHARS] for s in sents for i in range(0, len(s), CHUNK_CHARS)]  # hard-split run-ons
    out, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) > CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur} {u}".strip()
    return out + [cur] if cur else out


def to_chunks(docs: pd.DataFrame, rng) -> pd.DataFrame:
    rows = [(f"{d.id}#{i}", d.id, d.title, c) for d in docs.itertuples() for i, c in enumerate(chunk(d.text)) if len(c) >= 50]
    df = pd.DataFrame(rows, columns=["chunk_id", "doc_id", "title", "text"])
    df["tenant"] = rng.integers(0, 100, len(df))  # metadata for filtered-search tests
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--distractors", type=int, default=200_000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    wiki = pd.read_parquet(hf_hub_download("wikimedia/wikipedia", SHARD, repo_type="dataset"))
    wiki = wiki.sample(frac=1, random_state=0)
    picked = wiki[wiki.text.str.len().between(2000, 20000)].iloc[: a.docs]
    base = to_chunks(picked, rng)
    rest = wiki[~wiki.id.isin(picked.id) & (wiki.text.str.len() >= 500)]
    dist = to_chunks(rest.iloc[: a.distractors // 2], rng).iloc[: a.distractors]
    base.to_parquet("data/base.parquet")
    dist.to_parquet("data/distractors.parquet")
    print(f"base: {a.docs} docs, {len(base)} chunks | distractors: {len(dist)} chunks")


if __name__ == "__main__":
    main()
