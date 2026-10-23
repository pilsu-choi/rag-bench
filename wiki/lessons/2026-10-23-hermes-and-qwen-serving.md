---
type: Lesson
title: Running Hermes Agent and Qwen3.8 for a benchmark
description: Gotchas found while bringing up Hermes Agent headless against Qwen3.8-27B-NVFP4 on vLLM, and how each was fixed.
tags: [lesson, hermes, vllm, qwen, mcp]
timestamp: 2026-10-23T03:40:00Z
resource: ../../rag_bench/agentic/hermes.py
---

# Running Hermes Agent and Qwen3.8 for a benchmark

Hermes Agent is NousResearch/hermes-agent, pinned at commit `bf0203a`. It needs Python 3.14 and the
`[mcp]` extra, and lives outside the repo in its own venv.

## Qwen3.8-27B-NVFP4 on vLLM 0.31.0

- **Tool calls use the XML format.** The chat template emits
  `<tool_call><function=…><parameter=…>`, so the parser is `--tool-call-parser qwen3_coder` (hermes
  JSON would not parse). With it, a Korean question plus a `search_docs` tool returns a structured
  `tool_calls` entry.
- **It is a hybrid Mamba model.** The default `max_num_seqs` of 256 fails with "exceeds available Mamba
  cache blocks (175)". `--max-num-seqs 128` fixes it.
- **Decoding is about 11 tokens/s for one request** (NVFP4 on GB10). A 200-token answer takes about
  18 s, so the eval set size and concurrency are set by this.

## Hermes Agent

- **It refuses models with less than 64k context.** Qwen was first served with 32k, and every Hermes
  run failed inside the daemon with only a session id on stdout. The cause is in
  `$HERMES_HOME/logs/agent.log`. Qwen now runs with `--max-model-len 65536` at GPU memory 0.30
  (KV cache 275k tokens).
- **Config and MCP servers are read only when the background daemon starts.** The daemon runs one per
  `HERMES_HOME`. After any config change, kill `gateway.pid`; `hermes.py` does this automatically.
- **`--run-budget` is rejected in gateway mode.** Use `agent.run_budget_seconds` in config.yaml.
- **`--ignore-rules` also drops SOUL.md.** That is where the operator persona lives, so the RAG answer
  policy (answer only from the documents, cite `[chunk_id]`, otherwise refuse) never reached the model.
  The flag is no longer used. Rules files stay out because each question runs in an empty temp
  directory.
- **Default requests stream and send `reasoning_effort: medium`.** The logging proxy relays the SSE
  stream, rebuilds content, tool calls and usage for the log, and drops `reasoning_effort`, so thinking
  stays off as in every other track.
- **Remove the extras so that only our tools are visible.** Set `platform_toolsets.cli: [docsearch]`,
  `tools.tool_search.enabled: off`, MCP `resources: false, prompts: false`, memory and skill nudges off,
  and `auxiliary.title_generation.enabled: false`. Otherwise title generation adds one more LLM call per
  question.
- **Cost in practice:** one doc question took 5 LLM calls, and the prompt grew from 2.9k to 27k tokens,
  because the ~10k-character system prompt is resent every turn. A calculator question took about 15 s.

## Environment

- `pgx` is only an ssh alias and does not resolve for HTTP from WSL. Set `PGX_HOST` in the environment
  (never in the repo).
- One global tag on the logging proxy is not enough once several jobs share it: calls from the
  eval-set generation were tagged with a smoke test's tag. Each track now gets its own proxy port
  (`scripts/run-agentic.sh`).
