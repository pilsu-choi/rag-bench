#!/usr/bin/env bash
# Run every store at both scales, one DB container at a time (same 4 CPU / 6 GB limits).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
ready() {
  case $1 in
    pgvector) until docker exec bench-pgvector pg_isready -q 2>/dev/null; do sleep 2; done; sleep 5 ;;
    milvus) until curl -sf localhost:9091/healthz >/dev/null; do sleep 3; done ;;
    opensearch) until curl -sf localhost:9200/_cluster/health >/dev/null; do sleep 3; done ;;
  esac
}
for s in ${STORES:-faiss pgvector milvus opensearch}; do
  [ "$s" = faiss ] || { docker compose -f "docker/$s.yml" up -d --build; ready "$s"; }
  for scale in base full; do uv run python -m rag_bench.bench "$s" --scale "$scale"; done
  [ "$s" = faiss ] || docker compose -f "docker/$s.yml" down -v
done
uv run python -m rag_bench.model_latency
