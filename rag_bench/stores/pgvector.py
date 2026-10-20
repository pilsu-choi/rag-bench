"""PostgreSQL (ParadeDB image): pgvector HNSW + pg_search BM25 with lindera Korean tokenizer."""
import atexit
import numpy as np
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool


class Store:
    def __init__(self, dim: int):
        self.dim, self.ef = dim, 64
        with ConnectionPool("postgresql://postgres:bench@localhost:5432/postgres", min_size=1, max_size=64,
                            kwargs={"autocommit": True}, open=True) as p:
            with p.connection() as c:
                c.execute("CREATE EXTENSION IF NOT EXISTS vector")
                c.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
        self.pool = ConnectionPool("postgresql://postgres:bench@localhost:5432/postgres", min_size=1, max_size=64,
                                   kwargs={"autocommit": True}, configure=register_vector, open=True)
        atexit.register(self.pool.close)

    def build(self, ids, vecs, texts, tenants, m=16, ef_construction=200):
        with self.pool.connection() as c:
            c.execute("DROP TABLE IF EXISTS docs")
            c.execute(f"CREATE TABLE docs (id text PRIMARY KEY, tenant int, text text, emb vector({self.dim}))")
            with c.cursor().copy("COPY docs (id, tenant, text, emb) FROM STDIN (FORMAT BINARY)") as cp:
                cp.set_types(["text", "int4", "text", "vector"])
                for r in zip(ids, tenants, texts, np.asarray(vecs, dtype=np.float32)):
                    cp.write_row((str(r[0]), int(r[1]), r[2], r[3]))
            c.execute("SET maintenance_work_mem = '1GB'")
            c.execute("SET max_parallel_maintenance_workers = 4")
            c.execute(f"CREATE INDEX docs_emb ON docs USING hnsw (emb vector_ip_ops) WITH (m = {m}, ef_construction = {ef_construction})")
            c.execute("CREATE INDEX docs_tenant ON docs (tenant)")
            c.execute("CREATE INDEX docs_bm25 ON docs USING bm25 (id, (text::pdb.lindera(korean)))")
            c.execute("ANALYZE docs")

    def set_ef(self, ef):
        self.ef = ef

    def search(self, vec, k, tenant_lt=None):
        w, a = ("WHERE tenant < %s", (tenant_lt,)) if tenant_lt is not None else ("", ())
        with self.pool.connection() as c, c.transaction():
            c.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef)}")
            c.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            c.execute("SET LOCAL enable_seqscan = off")  # small tables: planner otherwise flips to exact sort at some ef
            return [r[0] for r in c.execute(f"SELECT id FROM docs {w} ORDER BY emb <#> %s LIMIT %s",
                                            (*a, np.asarray(vec, dtype=np.float32), k))]

    def bm25(self, query, k):
        with self.pool.connection() as c:
            return [r[0] for r in c.execute(
                "SELECT id FROM docs WHERE text ||| %s ORDER BY pdb.score(id) DESC LIMIT %s", (query, k))]

    def size_bytes(self):
        with self.pool.connection() as c:
            return c.execute("SELECT pg_total_relation_size('docs')").fetchone()[0]
