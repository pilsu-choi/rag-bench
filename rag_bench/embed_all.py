"""Embed base chunks, distractors and queries once (BGE-M3 on the PGX) -> data/emb_<name>.npy.

Usage: python -m rag_bench.embed_all [queries base distractors]
"""
from concurrent.futures import ThreadPoolExecutor
import sys, time
import numpy as np
from tqdm import tqdm
from rag_bench.common import DATA, embed, load

BATCH = 128

if __name__ == "__main__":
    names = sys.argv[1:] or ["queries", "base", "distractors"]
    for name in names:
        col = "query" if name == "queries" else "text"
        texts = load(name)[col].tolist()
        batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
        t = time.perf_counter()
        with ThreadPoolExecutor(8) as ex:
            vecs = list(tqdm(ex.map(embed, batches), total=len(batches), desc=name))
        np.save(f"{DATA}/emb_{name}.npy", np.vstack(vecs))
        print(f"{name}: {len(texts)} texts, {len(texts) / (time.perf_counter() - t):.0f} texts/s")
