"""Logging proxy for every LLM call in Part 2, Hermes included, so calls and tokens are counted alike.

    uv run python -m rag_bench.agentic.proxy        # listens on 127.0.0.1:8100, forwards to LLM_URL's /v1

Clients use base_url http://127.0.0.1:8100/v1. The harness tags the current question with
POST /_tag {"track": ..., "qid": ..., "rep": ...}; each forwarded request then appends one line
(tag, latency, usage, request messages, tool calls) to results/agentic/llm_calls.jsonl.
Runs are sequential (concurrency 1), so one global tag is enough.
"""
import json, os, time
import httpx, uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from rag_bench.common import LLM_MODEL, LLM_URL

UPSTREAM = LLM_URL.split("/v1/")[0] + "/v1"
KEY = os.environ.get("LLM_KEY", "")
# OpenRouter: pin one provider (no fallbacks) so every track hits the same backend; thinking off.
OR_EXTRA = {"provider": {"order": [os.environ["OR_PROVIDER"]], "allow_fallbacks": False},
            "reasoning": {"enabled": False}} if os.environ.get("OR_PROVIDER") else {}
LOG = "results/agentic/llm_calls.jsonl"
tag: dict = {}
client = httpx.AsyncClient(timeout=600)


async def set_tag(req: Request):
    tag.clear()
    tag.update(await req.json())
    return JSONResponse(tag)


def log(inp, ms, usage, content, tool_calls):
    rec = {**tag, "ts": time.time(), "ms": round(ms, 1), "usage": usage, "n_tools": len(inp.get("tools") or []),
           "messages": inp["messages"], "content": content, "tool_calls": tool_calls}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_delta(msg, delta):
    """Accumulate a streamed delta into {"content", "tool_calls"}."""
    msg["content"] += delta.get("content") or ""
    for tc in delta.get("tool_calls") or []:
        while len(msg["tool_calls"]) <= tc.get("index", 0):
            msg["tool_calls"].append({"function": {"name": "", "arguments": ""}})
        f, d = msg["tool_calls"][tc.get("index", 0)]["function"], tc.get("function") or {}
        f["name"] += d.get("name") or ""
        f["arguments"] += d.get("arguments") or ""


async def forward(req: Request):
    body, path = await req.body(), req.path_params["path"]
    headers = {"content-type": "application/json"} | ({"authorization": f"Bearer {KEY}"} if KEY else {})
    chat = path.endswith("chat/completions")
    inp = json.loads(body) if chat else {}
    if chat:  # every track runs with thinking off; Hermes sends reasoning_effort by default
        inp.pop("reasoning_effort", None)
        inp |= {"model": LLM_MODEL} | OR_EXTRA
        body = json.dumps(inp).encode()
    t = time.perf_counter()
    if chat and inp.get("stream"):
        inp.setdefault("stream_options", {"include_usage": True})
        up = await client.send(client.build_request("POST", f"{UPSTREAM}/{path}", content=json.dumps(inp), headers=headers),
                               stream=True)

        async def relay():
            msg, usage = {"content": "", "tool_calls": []}, None
            async for line in up.aiter_lines():
                yield line + "\n"
                if line.startswith("data: {"):
                    chunk = json.loads(line[6:])
                    usage = chunk.get("usage") or usage
                    for c in chunk.get("choices") or []:
                        merge_delta(msg, c.get("delta") or {})
            await up.aclose()
            if up.status_code == 200:
                log(inp, (time.perf_counter() - t) * 1000, usage, msg["content"], msg["tool_calls"] or None)

        return StreamingResponse(relay(), up.status_code, media_type="text/event-stream")
    r = await client.request(req.method, f"{UPSTREAM}/{path}", content=body or None, headers=headers)
    if chat and r.status_code == 200:
        msg = r.json()["choices"][0]["message"]
        log(inp, (time.perf_counter() - t) * 1000, r.json().get("usage"), msg.get("content"), msg.get("tool_calls"))
    return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))


app = Starlette(routes=[Route("/_tag", set_tag, methods=["POST"]),
                        Route("/v1/{path:path}", forward, methods=["GET", "POST"])])

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PROXY_PORT", 8100)), log_level="warning")
