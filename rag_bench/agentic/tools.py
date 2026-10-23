"""Tools shared by every Part 2 track: Korean BM25, dense, hybrid (RRF + rerank) search on the PGX
OpenSearch index, and a safe calculator.

The access filter is injected here (env TENANT_LT: the user may read chunks with tenant < TENANT_LT),
never passed by the model. Needs the index built by Part 1's opensearch adapter and `scripts/tunnel.sh`.
"""
import ast, operator, os
from opensearchpy import OpenSearch
from rag_bench.bench import RERANK_DEPTH, rrf
from rag_bench.common import embed, rerank
from rag_bench.stores.opensearch import IDX

_os = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], pool_maxsize=16, timeout=60)
DEPTH = RERANK_DEPTH


def _acl():
    t = os.environ.get("TENANT_LT")
    return [{"range": {"tenant": {"lt": int(t)}}}] if t else []


def _hits(query: dict, k: int) -> list[dict]:
    r = _os.search(index=IDX, body={"size": k, "_source": ["id", "text"], "query": query})
    return [{"id": h["_source"]["id"], "text": h["_source"]["text"]} for h in r["hits"]["hits"]]


def bm25(query: str, k: int = 5) -> list[dict]:
    return _hits({"bool": {"must": {"match": {"text": query}}, "filter": _acl()}}, k)


def dense(query: str, k: int = 5) -> list[dict]:
    knn = {"vector": embed([query])[0].tolist(), "k": k, "method_parameters": {"ef_search": 128}}
    if f := _acl():
        knn["filter"] = {"bool": {"filter": f}}
    return _hits({"knn": {"emb": knn}}, k)


def hybrid(query: str, k: int = 5) -> list[dict]:
    """BM25 and dense top-DEPTH fused with RRF, then the top DEPTH reranked by bge-reranker-v2-m3."""
    b, d = bm25(query, DEPTH), dense(query, DEPTH)
    by_id = {h["id"]: h for h in b + d}
    order = rrf([h["id"] for h in b], [h["id"] for h in d])[:DEPTH]
    scores = rerank(query, [by_id[i]["text"] for i in order])
    return [by_id[i] for _, i in sorted(zip(scores, order), reverse=True)[:k]]


SEARCH = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression: str) -> str:
    """Evaluate +, -, *, /, //, %, ** and parentheses on numbers; anything else is an error string."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        v = ev(ast.parse(expression.replace(",", "").replace("×", "*").replace("÷", "/"), mode="eval").body)
        return str(round(v, 6)) if isinstance(v, float) else str(v)
    except Exception as e:
        return f"error: {e}"


def fmt(hits: list[dict]) -> str:
    """Passages as the model sees them: numbered, with the chunk id as the citation key."""
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits) or "(검색 결과 없음)"
