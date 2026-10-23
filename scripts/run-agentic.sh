#!/usr/bin/env bash
# WSL side: run one Part 2 track over the given splits behind its own logging proxy, then judge it.
# usage: scripts/run-agentic.sh <track> <proxy-port> [split...]   (needs PGX_HOST and scripts/tunnel.sh)
# Each track gets its own proxy port, so tracks can run side by side without mixing up call tags.
# SMALL=1 limits main/acl to the `small` subset (Hermes tracks). WORKERS=N runs N questions of the track at once (calls are then attributed to questions by text).
set -euo pipefail
track=$1 port=$2; shift 2
splits=${*:-main acl consistency}
export PROXY_PORT=$port PROXY_URL=http://127.0.0.1:$port
uv run --extra agentic python -m rag_bench.agentic.proxy & proxy=$!
trap 'kill $proxy' EXIT
until curl -s -o /dev/null "$PROXY_URL/_tag" -d '{}'; do sleep 1; done
for s in $splits; do
  uv run --extra agentic python -W ignore -m rag_bench.agentic.run "$track" --split "$s" --workers "${WORKERS:-1}" ${SMALL:+--small}
  uv run --extra agentic python -W ignore -m rag_bench.agentic.judge "$track" --split "$s"
done
