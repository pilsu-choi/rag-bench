#!/usr/bin/env bash
# Run on the PGX: BGE-M3 embeddings (:8001) and bge-reranker-v2-m3 (:8002) with vLLM.
set -euo pipefail
serve() {  # name port model
  docker rm -f "$1" >/dev/null 2>&1 || true
  docker run -d --name "$1" --gpus all --ipc=host --restart no -p "$2:8000" \
    -v /mnt/ssd/models:/models:ro vllm/vllm-openai:v0.31.0 \
    "/models/$3" --served-model-name "$3" --runner pooling \
    --max-model-len 8192 --gpu-memory-utilization 0.15
}
serve embed 8001 bge-m3
serve rerank 8002 bge-reranker-v2-m3
