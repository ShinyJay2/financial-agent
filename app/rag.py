from .yahoo_client import get_price
from .ticker_map import find_ticker
from .vectorstore import hybrid_search, dense_collection
from .hyperclova_client import ask_hyperclova

SYSTEM_PROMPT = (
    "시스템: 당신은 금융 전문 AI 어시스턴트입니다. "
    "다음 문서 컨텍스트를 참고하여 답변하세요.\n"
)

def answer_question(question: str) -> str:
    # 1) 실시간 주가 의도
    if any(k in question for k in ("주가", "현재가", "시세")):
        for name in COMPANY_TICKERS:
            if name in question:
                ticker = find_ticker(name)
                if not ticker:
                    break
                price = get_price(ticker)
                if price is not None:
                    return f"{name}({ticker}) 현재 주가는 {price:,}원입니다."
                else:
                    return f"{ticker}의 주가를 조회할 수 없습니다."
        return "죄송합니다. 어떤 종목의 주가를 원하시는지 인식하지 못했습니다."

    # 2) 그 외: RAG retrieval
    top_ids = hybrid_search(question, k=5)
    docs = dense_collection.get(ids=top_ids)["documents"]
    context = "\n\n".join(f"[문서 {i+1}]\n{docs[i]}" for i in range(len(docs)))

    prompt = f"{SYSTEM_PROMPT}{context}\n\n사용자: {question}"
    return ask_hyperclova(prompt)
