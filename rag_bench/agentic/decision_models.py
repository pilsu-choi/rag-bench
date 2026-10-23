"""Laya and Kev as deciders. Both speak the same "System One" schema over HTTP:

    POST {url}/v1/systemone {"state": text, "questions": {"q": {"type": "choice", "instructions", "criteria"}}}
    -> {"answers": {"q": {"choice", "probabilities": {label: p}}}}

URLs come from env KEV_URL (kev.serve, Qwen3.5-4B + LoRA) and LAYA_URL (laya-serve, multilingual ckpt).
A yes/no label set {"yes", "no"} is asked as a `noul` question (probability of yes).
"""
import os
import httpx

INSTRUCTIONS = "입력에 가장 알맞은 선택지를 고르세요."
_http = httpx.Client(timeout=60)


class SystemOne:
    def __init__(self, name: str, url: str, model: str):
        self.name, self.url, self.model = name, url.rstrip("/"), model

    def _ask(self, text, q):
        r = _http.post(f"{self.url}/v1/systemone", json={"state": text, "model": self.model, "questions": {"q": q}})
        r.raise_for_status()
        return r.json()["answers"]["q"]

    def decide(self, text, labels):
        if set(labels) == {"yes", "no"}:
            p = float(self._ask(text, {"type": "noul", "instructions": labels["yes"] + "?"})["noul"])
            return ("yes", p) if p >= 0.5 else ("no", 1 - p)
        a = self._ask(text, {"type": "choice", "instructions": INSTRUCTIONS, "criteria": labels})
        return a["choice"], float(a["probabilities"][a["choice"]])


def get(name: str):
    if name == "kev":
        return SystemOne("kev", os.environ.get("KEV_URL", "http://pgx:8009"), "kev-latest")
    if name == "laya":
        return SystemOne("laya", os.environ.get("LAYA_URL", "http://pgx:8011"), "multilingual")
    raise SystemExit(f"unknown decider {name}")
