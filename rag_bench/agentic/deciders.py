"""Decision makers for the RAG workflow. Each one exposes

    decide(text, labels: {name: description}) -> (label, prob)

LLM (Qwen, JSON constrained to the label set), Laya and Kev (decision models), and Cascade
(a decision model whose low-confidence answers go to the LLM).
"""
from rag_bench.agentic import llm

PROMPT = """아래 입력에 가장 알맞은 선택지 하나를 고르세요.

입력:
{text}

선택지:
{options}"""


def options(labels: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in labels.items())


class LLM:
    name = "llm"

    def decide(self, text, labels):
        out = llm.ask(PROMPT.format(text=text, options=options(labels)), max_tokens=32,
                      schema={"type": "object", "properties": {"label": {"enum": list(labels)}}, "required": ["label"]})
        return out["label"], 1.0


class Cascade:
    """Use the primary decision if prob >= tau, otherwise ask the fallback (the LLM)."""

    def __init__(self, primary, fallback, tau: float):
        self.primary, self.fallback, self.tau = primary, fallback, tau
        self.name = f"{primary.name}-cascade{tau}"

    def decide(self, text, labels):
        label, p = self.primary.decide(text, labels)
        return (label, p) if p >= self.tau else self.fallback.decide(text, labels)


def get(name: str):
    if name == "llm":
        return LLM()
    from rag_bench.agentic import decision_models  # heavy deps only when needed
    return decision_models.get(name)
