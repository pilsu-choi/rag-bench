# AGENTS.md — rag-bench

Instructions for coding agents (Claude Code, etc.) working in this repository.
Read this whole file and [PRD.md](PRD.md) before writing code. When this file and a request conflict, ask.

## 1. What this repo is

Two benchmarks on one Korean RAG corpus (1,000 Korean Wikipedia articles, 9,069 chunks):

1. **Part 1 — retrieval** (`rag_bench/`, done 2026-10-20): FAISS, pgvector, Milvus and OpenSearch compared on
   ANN speed and retrieval quality. Results in [results/REPORT.md](results/REPORT.md).
2. **Part 2 — agentic RAG** (`rag_bench/agentic/`, in progress): a general agent runtime (Hermes Agent) with
   retrieval tools vs a RAG-specific workflow, and LLM tool calling vs decision models (Laya, Kev) for tool
   selection. Spec in [PRD.md](PRD.md).

## 2. Hard rules

1. **Change one variable per comparison.** Every Part 2 track uses the same corpus, retriever, answer LLM,
   prompts and eval set. If the answer LLM changes (e.g. to OpenRouter), re-run every track. Never mix
   endpoints in one table.
2. **All LLM calls go through the logging proxy**, Hermes included, so call counts and tokens are counted
   the same way for every track.
3. **Hermes runs isolated.** Each question gets a fresh state dir. Memory, skill creation, session search
   and built-in tools are off. If Hermes cannot run that way, stop and ask; do not substitute another
   framework or a hand-written loop and call it Hermes.
4. **Access filters are injected by the tool server**, never passed as a model-visible argument.
5. **No secrets, IP addresses or private host names in the repo or wiki.** Hosts come from env (`PGX_HOST`,
   `LLM_URL`, `LLM_MODEL`); keys from env (`OPENROUTER_API_KEY`). Use the ssh alias `pgx` in docs.
6. **Services with security off bind to loopback only.** OpenSearch on the PGX listens on
   `127.0.0.1:9200`; reach it with `scripts/tunnel.sh`.
7. **Don't guess external APIs.** Check Hermes, Laya and Kev behaviour against their docs or a run, and
   record what was verified in the wiki.

## 3. Layout

```
rag_bench/            Part 1: prep, synth, embed_all, bench, report, stores/
rag_bench/agentic/    Part 2: eval set, tools, MCP server, routers, workflow, runners, judge, report
scripts/              serve-retrieval.sh (Part 1), serve-agentic.sh (PGX, Part 2), tunnel.sh, run-all.sh
docker/               DB containers (OpenSearch image with nori)
data/                 gitignored; regenerate (Part 1: prep → synth → embed_all; Part 2: agentic evalset)
results/              Part 1 results; results/agentic/ for Part 2
wiki/                 work log and knowledge base (OKF), see §6
```

## 4. Environment

- Python ≥ 3.12 with `uv`. Run everything as `uv run python -m ...`. The system python has no deps.
- Part 2 dependencies live in the optional group `agentic` (`uv sync --extra agentic`), so Part 1 stays lean.
- Hermes Agent is an external tool with its own environment outside the repo, called as a subprocess.
- PGX (`ssh pgx`, aarch64, vLLM 0.31.0): Qwen3.8-27B-NVFP4 on :8000 (tool calling on, thinking off),
  bge-m3 on :8001, bge-reranker-v2-m3 on :8002, OpenSearch on loopback :9200. Start with
  `scripts/serve-agentic.sh` on the PGX. Only one session at a time starts or stops GPU containers.

## 5. Code style

- Minimal code. Reuse `rag_bench/common.py` and `rag_bench/stores/` instead of duplicating helpers; no
  overlapping classes or near-identical functions. One interface for decision makers:
  `decide(text, labels, descriptions) -> (label, prob)`.
- Module docstring says what the module does and how to run it; comments only where the why is not obvious.
- Raw per-question outputs are JSONL under `results/agentic/<track>/`; reports are generated from them, never
  edited by hand.

## 6. Workflow and wiki

- Git: branch from `dev` in a new worktree (`git worktree add ../rag-bench-<task> -b <branch> origin/dev`),
  commit there, merge back into `dev`, push `dev`, remove the worktree.
- Refresh `README.md` before every push.
- Record all work in `wiki/` in OKF v0.1:
  - Every page has YAML frontmatter with `type` (required), `title`, `description`, `tags`, `timestamp` (UTC),
    plus `resource` on pages about code.
  - New files are named `YYYY-MM-DD-<slug>.md`. `index.md` and `log.md` keep their names.
  - Add a dated UTC entry to `wiki/log.md` for each step, and link it to the page with details:
    `decisions/`, `runbooks/`, `research/`, `lessons/` or `issues/`.
  - Update the section's `index.md` when you add a page.
- Use sub-agents for independent parallel work (e.g. separate routers or tracks). Integration, review and
  commits stay with the orchestrating session.
