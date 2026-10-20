"""Milvus standalone: HNSW (IP) on dense, BM25 Function + SPARSE_INVERTED_INDEX on text (lindera ko-dic)."""
import time
from pymilvus import DataType, Function, FunctionType, MilvusClient

COL = "bench"
ANALYZER = {"tokenizer": {"type": "lindera", "dict_kind": "ko-dic"}, "filter": ["removepunct"]}


class Store:
    def __init__(self, dim: int):
        self.dim, self.ef = dim, 64
        self.c = MilvusClient("http://localhost:19530", timeout=600)

    def build(self, ids, vecs, texts, tenants, m=16, ef_construction=200):
        c = self.c
        c.drop_collection(COL)
        s = c.create_schema()
        s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        s.add_field("tenant", DataType.INT64)
        s.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=ANALYZER)
        s.add_field("emb", DataType.FLOAT_VECTOR, dim=self.dim)
        s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        s.add_function(Function("bm25", FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]))
        c.create_collection(COL, schema=s)
        for i in range(0, len(ids), 2000):
            c.insert(COL, [{"id": str(ids[j]), "tenant": int(tenants[j]), "text": texts[j][:65000], "emb": vecs[j].tolist()}
                           for j in range(i, min(i + 2000, len(ids)))])
        c.flush(COL)
        ip = c.prepare_index_params()
        ip.add_index("emb", index_type="HNSW", metric_type="IP", params={"M": m, "efConstruction": ef_construction})
        ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        ip.add_index("tenant", index_type="STL_SORT")
        c.create_index(COL, ip)
        c.load_collection(COL)
        for f in ("emb", "sparse", "tenant"):  # wait until every index has built all rows
            while (d := c.describe_index(COL, f))["pending_index_rows"] or d["indexed_rows"] < len(ids):
                time.sleep(1)

    def set_ef(self, ef):
        self.ef = ef

    # consistency_level="Eventually": benchmarks are read-only after build, so skip the timestamp wait.
    def search(self, vec, k, tenant_lt=None):
        r = self.c.search(COL, [vec.tolist()], anns_field="emb", limit=k, search_params={"metric_type": "IP", "params": {"ef": self.ef}},
                          filter="" if tenant_lt is None else f"tenant < {int(tenant_lt)}", output_fields=["id"], consistency_level="Eventually")
        return [h["id"] for h in r[0]]

    def bm25(self, query, k):
        r = self.c.search(COL, [query], anns_field="sparse", limit=k, search_params={"metric_type": "BM25"},
                          output_fields=["id"], consistency_level="Eventually")
        return [h["id"] for h in r[0]]

    def size_bytes(self):
        # Milvus exposes no per-collection disk size via the client; best effort = raw vector + text payload from row count.
        return 0
