# app/agent/routes.py

from fastapi import APIRouter, Query, HTTPException

from app.clients.hyperclova_client import ask_hyperclova
from app.agent.risk_agent import diagnose_risk
from app.utils.ticker_map import find_ticker_by_name

router = APIRouter()

@router.get(
    "/agent",
    summary="금융 질의 응답",
    response_model=dict
)
async def agent(
    question: str = Query(..., description="질문을 입력하세요"),
):
    """
    Ask HyperClova the given question and return its answer.
    """
    try:
        answer = ask_hyperclova(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/risk",
    summary="Risk 진단",
    response_model=dict
)
async def risk_diagnosis(
    symbol: str = Query(..., description="6자리 종목코드 또는 회사명")
):
    """
    Compute numeric risk metrics and summarize text-based risk factors for `symbol`.
    Accepts either a 6-digit ticker or a Korean company name.
    """
    # Normalize input to a ticker
    ticker = symbol if symbol.isdigit() else find_ticker_by_name(symbol)
    if not ticker:
        raise HTTPException(status_code=400, detail=f"Unknown company: {symbol}")

    try:
        result = diagnose_risk(ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
