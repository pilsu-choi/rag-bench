"""The RAG-specific workflow (tracks B0 and W-*), also served to Hermes as the RAG-layer tool (A2).

    intent ─┬─ chitchat ─────────────── LLM reply
            ├─ calc ─────────────────── LLM writes expression → calculator
            ├─ out_of_corpus ────────── refuse
            └─ doc_qa / multi_doc / doc_calc
                   → retrieval mode → search (multi_doc: one sub-query per entity)
                   → sufficiency yes/no → refuse, or LLM answer with [chunk_id] citations
                     (doc_calc: LLM writes expression from the passages → calculator)

`decider` makes the intent, mode and sufficiency decisions; the LLM only writes arguments and answers.
B0 skips every decision: always hybrid search, then answer.
"""
import time
from rag_bench.agentic import llm, tools
from rag_bench.agentic.catalog import INTENTS, MODES, REFUSAL

K = 5
YESNO = {"yes": "검색 결과에 질문의 답이 들어 있다", "no": "검색 결과로는 질문에 답할 수 없다"}

ANSWER = f"""검색 결과만 근거로 질문에 한국어로 간결하게 답하세요. 근거로 쓴 구절의 [id]를 문장 끝에 붙이세요.
검색 결과로 답할 수 없으면 정확히 "{REFUSAL}"라고만 답하세요.

검색 결과:
{{passages}}

질문: {{question}}"""
CHAT = "사용자의 말에 한국어로 짧고 친절하게 답하세요. 사실 정보가 필요한 질문이면 모른다고 하세요.\n\n{question}"
EXPR = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
EXPR_PROMPT = "질문에 답하는 계산식을 파이썬 산술식(숫자와 + - * / ** 괄호만)으로 쓰세요.\n{context}질문: {question}"
SUBQ = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
        "required": ["queries"]}
SUBQ_PROMPT = "질문에 나오는 대상마다 문서 검색용 짧은 검색어를 하나씩 만드세요.\n질문: {question}"


class Trace:
    """Per-question record: decisions (label, prob, ms), searches, answer."""

    def __init__(self):
        self.steps, self.retrieved = [], []

    def decide(self, decider, step, text, labels):
        t = time.perf_counter()
        label, p = decider.decide(text, labels)
        self.steps.append({"step": step, "label": label, "prob": round(float(p), 4),
                           "ms": round((time.perf_counter() - t) * 1000, 1)})
        return label

    def search(self, mode, query):
        t = time.perf_counter()
        hits = tools.SEARCH[mode](query, K)
        self.steps.append({"step": "search", "mode": mode, "query": query, "ids": [h["id"] for h in hits],
                           "ms": round((time.perf_counter() - t) * 1000, 1)})
        self.retrieved += [h for h in hits if h["id"] not in {r["id"] for r in self.retrieved}]
        return hits


def calc(question, context=""):
    try:
        expr = llm.ask(EXPR_PROMPT.format(context=context, question=question), schema=EXPR, max_tokens=128)["expression"]
    except ValueError as e:  # truncated JSON when the question is not arithmetic at all (a misroute)
        return "", f"error: {e}"
    return expr, tools.calculator(expr)


def run(question: str, decider=None) -> dict:
    """decider=None is B0. Returns {"answer", "intent", "steps", "retrieved"}."""
    tr = Trace()
    intent = tr.decide(decider, "intent", question, INTENTS) if decider else "doc_qa"
    if intent == "chitchat":
        answer = llm.ask(CHAT.format(question=question), max_tokens=128)
    elif intent == "calc":
        expr, val = calc(question)
        tr.steps.append({"step": "calculator", "expression": expr, "result": val})
        answer = val
    elif intent == "out_of_corpus":
        answer = REFUSAL
    else:
        mode = tr.decide(decider, "mode", question, MODES) if decider else "hybrid"
        queries = llm.ask(SUBQ_PROMPT.format(question=question), schema=SUBQ, max_tokens=128)["queries"] \
            if intent == "multi_doc" else [question]
        for q in queries or [question]:
            tr.search(mode, q)
        passages = tools.fmt(tr.retrieved)
        enough = tr.decide(decider, "sufficient", f"질문: {question}\n\n검색 결과:\n{passages}", YESNO) \
            if decider else "yes"
        if enough == "no":
            answer = REFUSAL
        elif intent == "doc_calc":
            expr, val = calc(question, f"검색 결과:\n{passages}\n\n")
            tr.steps.append({"step": "calculator", "expression": expr, "result": val})
            answer = val
        else:
            answer = llm.ask(ANSWER.format(passages=passages, question=question), max_tokens=512)
    return {"answer": answer, "intent": intent, "steps": tr.steps, "retrieved": [h["id"] for h in tr.retrieved],
            "passages": tools.fmt(tr.retrieved)}
