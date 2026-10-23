"""Build the Part 2 evaluation set -> data/agentic_eval.parquet (main 300 + acl 30 + toolscale 80).

    uv run --extra agentic python -m rag_bench.agentic.proxy &          # logging proxy, 127.0.0.1:8100
    bash scripts/tunnel.sh                                              # OpenSearch (out_of_corpus check)
    uv run --extra agentic python -m rag_bench.agentic.evalset

Qwen (via the proxy, temperature 0, concurrency 8) writes questions and reference answers; sampling is seeded.
Rows dropped by filters are counted and printed at the end.
"""
import collections, random, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from rag_bench.agentic import catalog as C, llm
from rag_bench.agentic.tools import bm25, calculator, dense
from rag_bench.common import load, vectors

SEED, OUT = 0, "data/agentic_eval.parquet"
N = dict(doc_qa=100, multi_doc=40, doc_calc=30, ooc_generic=25, ooc_entity=25, chitchat=40, calc=40, acl=30, per_tool=5)
drops, llm_s = collections.Counter(), [0.0]


def pmap(fn, items, workers=8):
    """Concurrent LLM calls; a failed call yields None."""
    def one(x):
        for _ in range(2):
            try:
                return fn(x)
            except Exception:
                pass
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, items))
    llm_s[0] += time.time() - t
    return out


def schema(**props):
    return {"type": "object", "properties": {k: v for k, v in props.items()}, "required": list(props)}


STR, BOOL = {"type": "string"}, {"type": "boolean"}
LIST = {"type": "array", "items": STR}
ITEMS = schema(items=LIST)


def row(split, intent, question, gold, ref, tool, **kw):
    return dict(split=split, intent=intent, question=question, gold_chunks=list(gold), ref_answer=ref, gold_tool=tool, **kw)


# ---------- doc_qa / acl ----------
def doc_qa(base, queries):
    q = queries.merge(base[["chunk_id", "title", "text", "tenant"]], on="chunk_id")
    pool, other = q[q.tenant < 50], q[q.tenant >= 50]
    qa = pool.sample(N["doc_qa"], random_state=SEED)
    ans = pmap(lambda r: llm.ask(f"다음 글을 읽고 질문에 한 문장 이내로 짧게 답하세요. 답만 출력하세요.\n\n[글]\n{r.text}\n\n[질문]\n{r.query}", max_tokens=200), list(qa.itertuples()))
    rows = [row("main", "doc_qa", r.query, [r.chunk_id], a, "search_documents") for r, a in zip(qa.itertuples(), ans) if a]
    drops["doc_qa: no answer"] += len(qa) - len(rows)
    acl = other.sample(N["acl"], random_state=SEED)
    rows += [row("acl", "doc_qa", r.query, [r.chunk_id], C.REFUSAL, "search_documents") for r in acl.itertuples()]
    return rows


# ---------- multi_doc ----------
def multi_doc(base, emb):
    ok = base.tenant.values < 50
    docs = sorted(set(base.doc_id[ok]))
    idx = {d: np.where((base.doc_id.values == d))[0] for d in docs}
    mean = np.stack([emb[idx[d]].mean(0) for d in docs])
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    sim = mean @ mean.T
    np.fill_diagonal(sim, -1)
    order, used, pairs = np.random.RandomState(SEED).permutation(len(docs)), set(), []
    for i in order:
        if i in used:
            continue
        j = next((j for j in np.argsort(-sim[i]) if j not in used), None)
        if j is None:
            continue
        used |= {i, j}
        pairs.append((i, j))
    def pick(a, b):  # a's eligible chunk closest to b's mean
        c = [k for k in idx[docs[a]] if ok[k]]
        return c[int(np.argmax(emb[c] @ mean[b]))]
    cand = [(pick(a, b), pick(b, a)) for a, b in pairs[: int(N["multi_doc"] * 1.6)]]
    P = """다음은 서로 다른 두 문서의 단락입니다.

[A: {ta}]
{xa}

[B: {tb}]
{xb}

두 단락의 내용을 모두 알아야만 답할 수 있는, 비교하거나 종합하는 자연스러운 한국어 질문 하나와 참조 답변을 만드세요.
- 질문만으로 대상이 분명해야 하며 "단락", "문서", "A", "B" 같은 표현은 쓰지 말 것
- 두 대상이 억지로 엮이면 ok를 false로 할 것
- 참조 답변은 두세 문장 이내"""
    outs = pmap(lambda p: llm.ask(P.format(ta=base.title[p[0]], xa=base.text[p[0]], tb=base.title[p[1]], xb=base.text[p[1]]),
                                  schema(question=STR, answer=STR, ok=BOOL), max_tokens=700), cand)
    rows = [row("main", "multi_doc", o["question"], [base.chunk_id[a], base.chunk_id[b]], o["answer"], "search_documents")
            for (a, b), o in zip(cand, outs) if o and o["ok"] and o["question"] and o["answer"] and not re.search("단락|문서", o["question"])]
    drops["multi_doc: llm not ok/banned/failed"] += len(cand) - len(rows)
    drops["multi_doc: over quota"] += max(0, len(rows) - N["multi_doc"])
    return rows[: N["multi_doc"]]


# ---------- doc_calc ----------
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:명|년|km|㎞|m|㎡|km²|원|%|퍼센트|개|회|만 ?명|억|세|톤|kg)")


def doc_calc(base, emb):
    ok = (base.tenant.values < 50) & (base.text.str.len().values >= 100) & base.text.str.contains(NUM).values
    ids = np.where(ok)[0]
    rs = np.random.RandomState(SEED)
    used, cand = set(), []
    for i in rs.permutation(ids):
        if i in used:
            continue
        s = emb[ids] @ emb[i]
        for j in ids[np.argsort(-s)]:
            if j != i and j not in used and base.doc_id[j] != base.doc_id[i]:
                used |= {i, j}
                cand.append((i, j))
                break
        if len(cand) >= N["doc_calc"] * 4:
            break
    P = """다음은 서로 다른 두 문서의 단락입니다.

[A: {ta}]
{xa}

[B: {tb}]
{xb}

두 단락에 나오는 구체적인 숫자(단위가 있는 값)를 하나씩 골라 더하기, 빼기, 곱하기, 나누기, 비율 같은 계산을 해야만 답이 나오는 한국어 질문 하나를 만드세요.
- 질문에는 단락의 숫자를 적지 말 것(숫자는 문서에서 찾아야 함). "단락", "문서"라는 표현도 쓰지 말 것
- expression: 계산식(숫자, + - * / ** 괄호만. 쉼표 없이)
- answer: 계산 결과 숫자만(단위 없이)
- 두 단락의 숫자를 함께 쓰는 자연스러운 질문이 없으면 ok를 false로 할 것"""
    outs = pmap(lambda p: llm.ask(P.format(ta=base.title[p[0]], xa=base.text[p[0]], tb=base.title[p[1]], xb=base.text[p[1]]),
                                  schema(question=STR, expression=STR, answer=STR, ok=BOOL), max_tokens=700), cand)
    rows = []
    for (a, b), o in zip(cand, outs):
        if not (o and o["ok"]):
            drops["doc_calc: llm not ok/failed"] += 1
            continue
        try:
            got, want = float(calculator(o["expression"])), float(o["answer"].replace(",", ""))
        except ValueError:
            drops["doc_calc: calculator/answer mismatch"] += 1
            continue
        if abs(got - want) > 1e-6 * max(1, abs(want)):
            drops["doc_calc: calculator/answer mismatch"] += 1
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", o["expression"])
        src = (base.text[a] + base.text[b]).replace(",", "")
        if len(nums) < 2 or sum(n in src for n in nums) < 2 or any(n in o["question"] for n in nums):
            drops["doc_calc: numbers not grounded / leaked in question"] += 1
            continue
        rows.append(row("main", "doc_calc", o["question"], [base.chunk_id[a], base.chunk_id[b]], calculator(o["expression"]), "search_documents"))
    drops["doc_calc: over quota"] += max(0, len(rows) - N["doc_calc"])
    return rows[: N["doc_calc"]]


# ---------- out_of_corpus ----------
GENERIC = ["내 개인 정보(나의 이름, 나이, 가족, 주소, 건강, 연락처, 비밀번호 등)를 AI가 알고 있다고 전제하고 묻는 질문",
           "미래 예측(다음 주 로또 번호, 내년 선거 결과, 10년 뒤 집값, 올림픽 우승국 등)을 요구하는 질문",
           "주관적 의견이나 취향(누가 더 잘생겼나, 어떤 음식이 최고인가, 어느 종교가 옳은가 등)을 요구하는 질문"]
TOPICS = ["고대 문명과 세계사", "유명 과학자와 발명", "세계의 국가와 수도", "전 세계 유명 인물(정치, 예술)", "스포츠 선수와 대회",
          "영화, 음악, 문학 작품과 작가", "자연과 동식물", "천문학과 우주", "한국의 역사 인물", "유명 기업과 브랜드",
          "세계의 유명 건축물과 랜드마크", "철학자와 사상", "의학과 질병", "전쟁과 조약"]


def listing(prompt):
    return llm.ask(prompt, ITEMS, max_tokens=1500)["items"]


def out_of_corpus(base):
    P = "한국어로 AI 비서에게 묻는 다음 유형의 질문을 서로 다르게 {n}개 만드세요.\n유형: {t}\n실시간 시세, 날씨, 코드, 뉴스, 환율, 택배 관련 질문은 제외합니다."
    g = pmap(lambda t: listing(P.format(n=12, t=t)), GENERIC)
    generic = list(dict.fromkeys(q.strip() for x in g if x for q in x))
    random.Random(SEED).shuffle(generic)
    rows = [row("main", "out_of_corpus", q, [], C.REFUSAL, "refuse") for q in generic[: N["ooc_generic"]]]
    drops["ooc generic: over quota"] += max(0, len(generic) - N["ooc_generic"])
    # entity questions
    EP = "한국어 위키백과에 항목이 있을 법한 유명한 대상에 대해, 위키백과식 사실 확인 질문을 {n}개 만드세요.\n분야: {t}\n질문마다 질문의 핵심 대상 이름(subject)을 함께 적으세요. 대상은 서로 달라야 합니다."
    S = schema(items={"type": "array", "items": schema(subject=STR, question=STR)})
    e = pmap(lambda t: llm.ask(EP.format(n=8, t=t), S, max_tokens=1500)["items"], TOPICS)
    seen, ents = set(), []
    titles = list(base.title)
    for x in e:
        for it in x or []:
            s = it["subject"].replace(" ", "")
            if s in seen:
                continue
            seen.add(s)
            if any(s in t.replace(" ", "") or t.replace(" ", "") in s for t in titles) or any(t in it["question"] for t in titles if len(t) >= 3):
                drops["ooc entity: subject in corpus titles"] += 1
                continue
            ents.append(it["question"])
    V = """질문: {q}

아래 검색 결과 중 이 질문에 답할 수 있는 내용이 있습니까? 질문의 대상에 대한 정보가 직접 들어 있어야 합니다.

{ctx}"""
    def check(q):
        hits = {h["id"]: h["text"] for h in bm25(q, 10) + dense(q, 10)}
        ctx = "\n\n".join(f"[{i}] {t[:600]}" for i, t in hits.items())
        return not llm.ask(V.format(q=q, ctx=ctx), schema(answerable=BOOL), max_tokens=50)["answerable"]
    clean = pmap(check, ents)
    drops["ooc entity: answerable from retrieval"] += sum(c is False for c in clean)
    drops["ooc entity: check failed"] += sum(c is None for c in clean)
    ents = [q for q, c in zip(ents, clean) if c]
    random.Random(SEED).shuffle(ents)
    drops["ooc entity: over quota"] += max(0, len(ents) - N["ooc_entity"])
    return rows + [row("main", "out_of_corpus", q, [], C.REFUSAL, "search_documents") for q in ents[: N["ooc_entity"]]]


# ---------- chitchat / calc / toolscale ----------
KINDS = ["인사(아침, 점심, 저녁, 처음 만났을 때)", "기분과 감정 표현(피곤함, 신남, 우울함, 심심함)", "AI에게 자기소개나 근황을 묻는 말",
         "감사와 칭찬, 사과", "작별 인사와 덕담", "가벼운 농담이나 끝말잇기 같은 놀이 제안", "응원과 위로가 필요한 말"]


def chitchat():
    P = "한국어 일상 대화체로 AI 비서에게 건네는 가벼운 잡담을 서로 다르게 {n}개 만드세요.\n유형: {t}\n정보 검색, 날씨, 일정, 계산이 필요한 말은 제외합니다. 말투(반말, 존댓말, 이모티콘)도 다양하게."
    out = pmap(lambda t: listing(P.format(n=9, t=t)), KINDS)
    qs = list(dict.fromkeys(q.strip() for x in out if x for q in x))
    random.Random(SEED).shuffle(qs)
    drops["chitchat: over quota"] += max(0, len(qs) - N["chitchat"])
    return [row("main", "chitchat", q, [], "", "no_tool") for q in qs[: N["chitchat"]]]


def kor(n: int) -> str:
    """12000 -> '1만 2천'"""
    man, rest = divmod(n, 10000)
    return " ".join(([f"{man}만"] if man else []) + ([f"{rest // 1000}천"] if rest // 1000 else []) + ([str(rest % 1000)] if rest % 1000 else []))


def calc():
    rng, rows, seen = random.Random(SEED), [], set()
    def mk():
        t = rng.randrange(8)
        a, b = rng.randint(12, 9999), rng.randint(2, 99)
        if t == 0: return f"{a} 곱하기 {b}은?", a * b
        if t == 1: return f"{a} 더하기 {(c := rng.randint(100, 9999))}은?", a + c
        if t == 2: return f"{a + b * 100}에서 {a}을 빼면 얼마야?", b * 100
        if t == 3: return f"{a * b}를 {b}로 나누면?", a
        if t == 4: return f"{(c := rng.randint(2, 20))}의 {(e := rng.randint(2, 4))}제곱은?", c ** e
        if t == 5:
            p, x = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75]), rng.randint(2, 99) * 100
            return f"{x}의 {p}%는 얼마야?", x * p / 100
        if t == 6:
            x, y = rng.randint(1, 9) * 1000 + rng.randint(0, 9) * 10000, rng.randint(1, 9) * 1000
            return f"{kor(x)}에서 {kor(y)}을 빼면?", x - y
        x, y, z = rng.randint(2, 50), rng.randint(2, 50), rng.randint(2, 20)
        return f"({x} + {y}) × {z}는?", (x + y) * z
    while len(rows) < N["calc"]:
        q, v = mk()
        if q not in seen:
            seen.add(q)
            rows.append(row("main", "calc", q, [], str(int(v) if float(v).is_integer() else v), "calculator"))
    return rows


def toolscale():
    P = "회사 직원이 업무 도구 '{k}'({d})를 써야만 해결되는 요청을 한국어로 {n}개 만드세요.\n서로 다른 상황과 말투로, 구체적인 날짜·이름·번호 같은 세부 정보를 넣어 자연스럽게. 다른 도구가 아니라 이 도구가 필요한 요청이어야 합니다."
    items = list(C.DISTRACTOR_TOOLS.items())
    out = pmap(lambda kd: listing(P.format(k=kd[0], d=kd[1], n=8)), items)
    rows = []
    for (k, _), x in zip(items, out):
        qs = list(dict.fromkeys(q.strip() for q in x or []))
        drops["toolscale: over quota/short"] += len(qs) - N["per_tool"] if len(qs) >= N["per_tool"] else N["per_tool"] - len(qs)
        rows += [row("toolscale", k, q, [], "", k) for q in qs[: N["per_tool"]]]
    return rows


# ---------- assemble ----------
def stratify(df, total=50, floor=5):
    """Consistency flags: proportional by intent over main rows, at least `floor` each."""
    cnt = df[df.split == "main"].intent.value_counts()
    k = {i: max(floor, round(total * c / cnt.sum())) for i, c in cnt.items()}
    while sum(k.values()) != total:  # trim or pad the largest stratum
        k[max(k, key=k.get)] += 1 if sum(k.values()) < total else -1
    flag = set()
    for i, n in k.items():
        flag |= set(df[(df.split == "main") & (df.intent == i)].sample(n, random_state=SEED).index)
    return df.index.isin(flag)


def main():
    base, queries, emb = load("base"), load("queries"), vectors("base")
    llm.tag(track="evalset")
    rows = []
    for name, f in [("doc_qa+acl", lambda: doc_qa(base, queries)), ("multi_doc", lambda: multi_doc(base, emb)), ("doc_calc", lambda: doc_calc(base, emb)),
                    ("out_of_corpus", lambda: out_of_corpus(base)), ("chitchat", chitchat), ("calc", calc), ("toolscale", toolscale)]:
        t = time.time()
        rows += (r := f())
        print(f"{name}: {len(r)} rows ({time.time() - t:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["qid"] = [f"{('acl' if s == 'acl' else i)}-{n:03d}" for s, i, n in zip(df.split, df.intent, df.groupby(["split", "intent"]).cumcount() + 1)]
    df["consistency"] = stratify(df)
    df = df[["qid", "split", "intent", "question", "gold_chunks", "ref_answer", "gold_tool", "consistency"]]
    assert df.qid.is_unique
    df.to_parquet(OUT)
    print(df.groupby(["split", "intent"]).size().to_string(), f"\nconsistency={int(df.consistency.sum())}  total={len(df)}")
    for i, g in df.groupby("intent"):
        print(f"\n== {i}")
        for r in g.sample(min(2, len(g)), random_state=SEED).itertuples():
            print(f"  [{r.qid}] {r.question}\n    -> {r.ref_answer!r} gold={list(r.gold_chunks)} tool={r.gold_tool}")
    print("\nDropped:", dict(drops), f"\nLLM wall time: {llm_s[0]:.0f}s")


if __name__ == "__main__":
    main()
