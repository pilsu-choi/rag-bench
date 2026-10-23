#!/usr/bin/env bash
# WSL side: forward OpenSearch (9200), Kev (8009) and Laya (8011) from the PGX (ssh alias pgx), all loopback-only there.
set -euo pipefail
for p in 9200 8009 8011; do
  curl -s -o /dev/null -m 2 "localhost:$p" && continue
  ssh -fN -o ExitOnForwardFailure=yes -L "$p:localhost:$p" pgx && echo "tunnel $p opened"
done
