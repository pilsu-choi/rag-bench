"""Chat client for the in-house tracks. Every call goes through the logging proxy (rag_bench.agentic.proxy)."""
import json, os
import httpx
from rag_bench.common import LLM_MODEL

PROXY = os.environ.get("PROXY_URL", "http://127.0.0.1:8100")
_http = httpx.Client(timeout=600)


def tag(**kw):
    """Attribute the following LLM calls to one question of one track."""
    _http.post(f"{PROXY}/_tag", json=kw).raise_for_status()


def chat(messages: list[dict], tools: list[dict] | None = None, schema: dict | None = None, max_tokens=512) -> dict:
    """One completion at temperature 0; returns the assistant message (content and/or tool_calls)."""
    body = {"model": LLM_MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r = _http.post(f"{PROXY}/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def ask(prompt: str, schema: dict | None = None, max_tokens=512, system: str | None = None):
    """Single-turn helper: text, or the parsed JSON object when a schema is given."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    out = chat(msgs, schema=schema, max_tokens=max_tokens)["content"]
    return json.loads(out) if schema else out.strip()
