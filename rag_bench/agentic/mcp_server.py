"""MCP (stdio) server that gives Hermes the same tools as the in-house tracks.

    python -m rag_bench.agentic.mcp_server a1   # search_bm25, search_dense, search_hybrid, calculator
    python -m rag_bench.agentic.mcp_server a2   # ask_knowledge_base (the RAG workflow), calculator

The access filter comes from env TENANT_LT (see tools.py), never from a tool argument. The RAG layer
in a2 uses the decider named by env RAG_DECIDER (default llm).
"""
import os, sys
from mcp.server.mcpserver import MCPServer
from rag_bench.agentic import deciders, tools, workflow

srv = MCPServer("docsearch")


def calculator(expression: str) -> str:
    """산술식을 계산합니다(+ - * / ** 괄호). 예: "1234*56"."""
    return tools.calculator(expression)


def search_bm25(query: str) -> str:
    """위키백과 문서를 키워드(BM25)로 검색해 상위 5개 구절을 [chunk_id]와 함께 돌려줍니다. 고유명사·숫자가 있는 질문에 적합합니다."""
    return tools.fmt(tools.bm25(query))


def search_dense(query: str) -> str:
    """위키백과 문서를 의미(임베딩)로 검색해 상위 5개 구절을 [chunk_id]와 함께 돌려줍니다. 표현이 다른 질문에 적합합니다."""
    return tools.fmt(tools.dense(query))


def search_hybrid(query: str) -> str:
    """위키백과 문서를 키워드+의미로 검색하고 재정렬해 상위 5개 구절을 [chunk_id]와 함께 돌려줍니다. 가장 정확하지만 가장 느립니다."""
    return tools.fmt(tools.hybrid(query))


def ask_knowledge_base(question: str) -> str:
    """사내 지식베이스(위키백과 문서)에 질문하면 근거 구절과 출처 [chunk_id]가 붙은 답변을 돌려줍니다.
    문서로 답할 수 없으면 "문서에서 찾을 수 없습니다"를 돌려줍니다."""
    r = workflow.run(question, deciders.get(os.environ.get("RAG_DECIDER", "llm")))
    return f"답변: {r['answer']}\n\n근거:\n{r['passages']}"


TOOLS = {"a1": [search_bm25, search_dense, search_hybrid, calculator], "a2": [ask_knowledge_base, calculator]}

if __name__ == "__main__":
    for fn in TOOLS[sys.argv[1]]:
        srv.tool()(fn)
    srv.run()
