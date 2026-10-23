#!/usr/bin/env bash
# Run on the PGX: Qwen LLM :8000, bge-m3 :8001, reranker :8002, OpenSearch (loopback :9200); idempotent, sequential.
set -euo pipefail
IMG=vllm/vllm-openai:v0.31.0
wait_for() {  # name url
  for _ in $(seq 180); do
    curl -sf "$2" >/dev/null && { echo "$1 ready"; return; }
    docker ps -q -f name="^$1$" | grep -q . || { docker logs "$1" 2>&1 | tail -30; exit 2; }
    sleep 10
  done
  docker logs "$1" 2>&1 | tail -30; exit 3
}
vllm() {  # name port gmu model-and-args...
  local n=$1 p=$2 g=$3; shift 3
  curl -sf "localhost:$p/v1/models" >/dev/null 2>&1 && { echo "$n up"; return; }
  docker rm -f "$n" >/dev/null 2>&1 || true
  docker run -d --name "$n" --gpus all --ipc=host --restart no -p "$p:8000" -v /mnt/ssd/models:/models:ro "$IMG" \
    "$@" --gpu-memory-utilization "$g" >/dev/null
  wait_for "$n" "localhost:$p/v1/models"
}
flock -n ~/folio-m1-bench/gpu.lock true 2>/dev/null || [ ! -e ~/folio-m1-bench/gpu.lock ] || { echo "GPU lock held"; exit 1; }
vllm llm 8000 0.30 --model /models/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b --max-model-len 65536 --max-num-seqs 128 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
vllm embed 8001 0.08 --model /models/bge-m3 --served-model-name bge-m3 --runner pooling --max-model-len 8192
vllm rerank 8002 0.08 --model /models/bge-reranker-v2-m3 --served-model-name bge-reranker-v2-m3 --runner pooling --max-model-len 8192
if ! curl -sf localhost:9200 >/dev/null 2>&1; then
  docker image inspect bench-opensearch:3.9.0-nori >/dev/null 2>&1 || docker build -t bench-opensearch:3.9.0-nori ~/agentic-opensearch
  docker rm -f agentic-opensearch >/dev/null 2>&1 || true
  docker run -d --name agentic-opensearch --restart no -p 127.0.0.1:9200:9200 \
    -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
    -e OPENSEARCH_JAVA_OPTS="-Xms6g -Xmx6g" -e bootstrap.memory_lock=true \
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 bench-opensearch:3.9.0-nori >/dev/null
  wait_for agentic-opensearch localhost:9200
fi
