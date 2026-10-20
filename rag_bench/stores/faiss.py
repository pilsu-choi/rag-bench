"""FAISS HNSW in-process: no persistence, no BM25; filtering via IDSelector."""
import faiss, numpy as np


class Store:
    def __init__(self, dim: int):
        self.dim = dim
        faiss.omp_set_num_threads(4)  # same CPU budget as the DB containers

    def build(self, ids, vecs, texts, tenants, m=16, ef_construction=200):
        self.ids, self.tenants = np.asarray(ids), np.asarray(tenants)
        self.index = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self.ef, self.sel = 64, {}

    def set_ef(self, ef):
        self.ef = ef

    def search(self, vec, k, tenant_lt=None):
        p = faiss.SearchParametersHNSW(efSearch=self.ef)
        if tenant_lt is not None:
            if tenant_lt not in self.sel:  # cache: the selector is per filter, not per query
                self.sel[tenant_lt] = faiss.IDSelectorBatch(np.flatnonzero(self.tenants < tenant_lt))
            p.sel = self.sel[tenant_lt]
        _, I = self.index.search(vec[None].astype(np.float32), k, params=p)
        return self.ids[I[0][I[0] >= 0]].tolist()

    def bm25(self, query, k):
        raise NotImplementedError

    def size_bytes(self):
        return faiss.serialize_index(self.index).nbytes
