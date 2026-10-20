"""OpenSearch k-NN (faiss HNSW, innerproduct) + nori BM25; 1 shard, force-merged to 1 segment."""
from opensearchpy import OpenSearch, helpers

IDX = "bench"


class Store:
    def __init__(self, dim: int):
        self.dim, self.ef = dim, 64
        self.c = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], pool_maxsize=64, timeout=600)

    def build(self, ids, vecs, texts, tenants, m=16, ef_construction=200):
        self.c.indices.delete(index=IDX, ignore_unavailable=True)
        self.c.indices.create(index=IDX, body={
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True, "refresh_interval": "-1"},
                         "analysis": {"analyzer": {"ko": {"type": "custom", "tokenizer": "nori_tokenizer",
                                                          "filter": ["nori_part_of_speech", "nori_readingform", "lowercase"]}}}},
            "mappings": {"properties": {
                "id": {"type": "keyword"}, "tenant": {"type": "integer"}, "text": {"type": "text", "analyzer": "ko"},
                "emb": {"type": "knn_vector", "dimension": self.dim, "method": {
                    "name": "hnsw", "engine": "faiss", "space_type": "innerproduct",
                    "parameters": {"m": m, "ef_construction": ef_construction}}}}}})
        helpers.bulk(self.c, ({"_index": IDX, "_id": str(i), "_source": {"id": str(i), "text": t, "tenant": int(n), "emb": v.tolist()}}
                              for i, v, t, n in zip(ids, vecs, texts, tenants)), chunk_size=500, request_timeout=600)
        self.c.indices.refresh(index=IDX)
        self.c.indices.forcemerge(index=IDX, max_num_segments=1, request_timeout=3600)  # single segment => single HNSW graph
        self.c.indices.put_settings(index=IDX, body={"index": {"refresh_interval": "1s"}})
        self.c.transport.perform_request("GET", f"/_plugins/_knn/warmup/{IDX}", params={"request_timeout": 3600})

    def set_ef(self, ef):
        self.ef = ef

    def _ids(self, q, k):
        r = self.c.search(index=IDX, body={"size": k, "_source": False, "docvalue_fields": ["id"], "query": q})
        return [h["fields"]["id"][0] for h in r["hits"]["hits"]]

    def search(self, vec, k, tenant_lt=None):
        e = {"vector": vec.tolist(), "k": k, "method_parameters": {"ef_search": self.ef}}
        if tenant_lt is not None:
            e["filter"] = {"range": {"tenant": {"lt": int(tenant_lt)}}}
        return self._ids({"knn": {"emb": e}}, k)

    def bm25(self, query, k):
        return self._ids({"match": {"text": query}}, k)

    def size_bytes(self):
        return self.c.indices.stats(index=IDX, metric="store")["_all"]["primaries"]["store"]["size_in_bytes"]
