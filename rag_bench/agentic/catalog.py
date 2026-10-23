"""Label sets shared by the eval set, the routers and the judge: intents, retrieval modes, and the
distractor business tools used in the tool-count sweep (5 / 10 / 20 tools)."""

INTENTS = {
    "doc_qa": "위키백과 문서에서 사실 하나를 찾아 답하는 질문",
    "multi_doc": "두 개 이상의 대상(인물, 장소, 사건 등)을 문서에서 각각 찾아 비교하거나 종합하는 질문",
    "doc_calc": "문서에서 숫자를 찾은 뒤 그 숫자로 계산해야 하는 질문",
    "out_of_corpus": "개인 정보, 미래 예측, 주관적 의견처럼 백과사전 문서로 답할 수 없는 질문, 또는 문서에 없는 대상에 대한 질문",
    "chitchat": "인사, 감정 표현, 잡담처럼 정보 검색이 필요 없는 대화",
    "calc": "문서 없이 주어진 숫자만으로 계산하는 산술 질문",
}
SEARCH_INTENTS = {"doc_qa", "multi_doc", "doc_calc"}
CALC_INTENTS = {"calc", "doc_calc"}
REFUSAL = "문서에서 찾을 수 없습니다"

MODES = {
    "bm25": "고유명사, 숫자, 정확한 표현이 들어 있어 키워드가 그대로 맞으면 되는 질문",
    "dense": "표현을 바꿔 말했거나 의미로 찾아야 하는 질문",
    "hybrid": "키워드와 의미가 둘 다 중요하거나 판단이 어려운 질문(가장 정확하지만 가장 느림)",
}

# The RAG tools as one router sees them: choosing a tool == choosing an action.
RAG_TOOLS = {
    "search_documents": "위키백과 문서 검색. 사실 확인, 인물·장소·사건·역사에 대한 질문",
    "calculator": "산술 계산(사칙연산, 거듭제곱)",
    "no_tool": "도구 없이 바로 답함. 인사, 잡담",
    "refuse": "어떤 도구로도 답할 수 없는 질문(개인 정보, 미래 예측, 주관적 의견)",
}

DISTRACTOR_TOOLS = {
    "calendar_lookup": "사내 일정과 회의 일정 조회",
    "send_email": "사내 메일 작성과 발송",
    "approval_submit": "전자결재 상신(휴가, 지출, 구매 결재)",
    "weather": "날씨와 기온 예보 조회",
    "hr_leave_balance": "남은 연차와 휴가 일수 조회",
    "meeting_room_booking": "회의실 예약",
    "expense_report": "법인카드 사용 내역과 경비 정산",
    "it_helpdesk_ticket": "IT 장애 신고와 헬프데스크 티켓 생성",
    "translate": "문장 번역",
    "org_chart": "임직원 연락처와 조직도 조회",
    "stock_price": "주식과 코인 실시간 시세 조회",
    "shipping_tracking": "택배 배송 조회",
    "currency_convert": "환율 조회와 통화 환산",
    "code_runner": "파이썬 코드 실행",
    "reminder": "알림과 리마인더 등록",
    "news_search": "최신 뉴스 검색",
}


def tool_set(n: int) -> dict:
    """RAG tools plus the first n - len(RAG_TOOLS) distractors (n = 4, 10 or 20)."""
    return RAG_TOOLS | dict(list(DISTRACTOR_TOOLS.items())[: max(0, n - len(RAG_TOOLS))])
