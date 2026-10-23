#!/usr/bin/env bash
# WSL side: forward localhost:9200 to the PGX (ssh alias pgx) OpenSearch (loopback-only there) if not already open.
set -euo pipefail
curl -sf localhost:9200 >/dev/null 2>&1 && { echo "tunnel up"; exit 0; }
ssh -fN -o ExitOnForwardFailure=yes -L 9200:localhost:9200 pgx
echo "tunnel opened"
