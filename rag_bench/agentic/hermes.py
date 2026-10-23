"""Tracks A1/A2: Hermes Agent (external install, pinned commit) with our MCP server as its only toolset.

Hermes lives outside the repo (HERMES_BIN, default /home/pilsu/tools/hermes-agent/.venv/bin/hermes).
One HERMES_HOME per (track, split, access filter) under results/agentic/.hermes/; memory, skills, nudges,
title generation and every built-in toolset are off, and each `-q` call is a new session, so questions
do not see each other. The LLM is reached through the logging proxy. When the config changes, the
background daemon is restarted, because Hermes reads config and MCP servers only at daemon start.
"""
import json, os, re, signal, subprocess, tempfile
import yaml
from rag_bench.agentic.llm import PROXY
from rag_bench.common import LLM_MODEL

BIN = os.environ.get("HERMES_BIN", "/home/pilsu/tools/hermes-agent/.venv/bin/hermes")
REPO = os.path.abspath(".")
ENV_KEYS = ["PGX_HOST", "TENANT_LT", "RAG_DECIDER", "PROXY_URL", "KEV_URL", "LAYA_URL", "LLM_MODEL"]
CHUNK = re.compile(r"\[(\d+#\d+)\]")
_ready: set = set()


def config(track: str) -> dict:
    env = {k: os.environ[k] for k in ENV_KEYS if k in os.environ} | {"PROXY_URL": PROXY}
    return {
        "model": {"default": LLM_MODEL, "provider": "custom", "base_url": f"{PROXY}/v1", "api_key": "EMPTY"},
        "mcp_servers": {"docsearch": {
            "command": "uv", "args": ["run", "--directory", REPO, "--extra", "agentic", "python", "-m",
                                      "rag_bench.agentic.mcp_server", track],
            "cwd": REPO, "env": env, "tools": {"resources": False, "prompts": False}}},
        "memory": {"memory_enabled": False, "user_profile_enabled": False, "nudge_interval": 0},
        "skills": {"creation_nudge_interval": 0},
        "platform_toolsets": {"cli": ["docsearch"]},
        "tools": {"tool_search": {"enabled": "off"}},
        "auxiliary": {"title_generation": {"enabled": False}},
    }


def home(track: str) -> str:
    h = os.path.abspath(f"results/agentic/.hermes/{track}-tenant{os.environ.get('TENANT_LT', 'all')}")
    if h not in _ready:
        os.makedirs(h, exist_ok=True)
        cfg = os.path.join(h, "config.yaml")
        new = yaml.safe_dump(config(track), allow_unicode=True)
        if not os.path.exists(cfg) or open(cfg).read() != new:
            open(cfg, "w").write(new)
            stop(h)
        _ready.add(h)
    return h


def stop(h: str):
    pid = os.path.join(h, "gateway.pid")
    if os.path.exists(pid):
        try:
            os.kill(int(open(pid).read().split()[0]), signal.SIGTERM)
        except (ValueError, ProcessLookupError):
            pass


def run(question: str, track: str) -> dict:
    h = home(track)
    with tempfile.TemporaryDirectory() as cwd:  # empty cwd: no AGENTS.md or rules get injected
        p = subprocess.run([BIN, "--ignore-rules", "chat", "-Q", "--format", "stream-json", "--max-turns", "10",
                            "--run-budget", "300", "-q", question], cwd=cwd, env=os.environ | {"HERMES_HOME": h},
                           capture_output=True, text=True, timeout=400)
    events = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    calls = [{"step": "tool", "name": e["name"].split("__")[-1], "input": e.get("input")}
             for e in events if e.get("type") == "tool_use"]
    outputs = [str(e.get("output", "")) for e in events if e.get("type") == "tool_result"]
    last = max((i for i, e in enumerate(events) if e.get("type") == "tool_result"), default=-1)
    answer = "".join(e.get("text", "") for e in events[last + 1:] if e.get("type") == "text")  # final turn only
    final = next((e for e in events if e.get("type") == "result"), {})
    retrieved = list(dict.fromkeys(CHUNK.findall("\n".join(outputs))))
    out = {"answer": answer.strip(), "steps": calls, "retrieved": retrieved, "passages": "\n\n".join(outputs),
           "hermes": {k: final.get(k) for k in ("exit_code", "tokens", "duration_ms")}}
    if p.returncode or not events:
        out["error"] = (p.stderr or p.stdout)[-500:]
    return out
