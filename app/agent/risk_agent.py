# app/agent/risk_agent.py

from datetime import date, timedelta
from typing import Dict, Any

from app.ingestion.dart_fss_client import (
    corp_code_for,
    get_risk_factors,
    extract_financial_statements,
)
from app.rag_pipeline import RAGPipeline
from app.risk.volatility  import realized_volatility
from app.risk.drawdown    import max_drawdown
from app.risk.implied_vol import implied_volatility
from app.ingestion.yahoo_client import get_recent_news   # you’ll need to implement this
from app.ingestion.krx_client import get_market_ohlcv_by_date


def diagnose_risk(ticker: str) -> Dict[str, Any]:
    # ─── 1) Numeric market metrics ─────────────────────────────
    rv = realized_volatility(ticker)
    md = max_drawdown(ticker)
    iv = implied_volatility(ticker)

    # ─── 2) Recent price snapshot (last 5 days) ────────────────
    today = date.today().strftime("%Y%m%d")
    week_ago = (date.today() - timedelta(days=7)).strftime("%Y%m%d")
    ohlcv = get_market_ohlcv_by_date(fromdate=week_ago, todate=today, ticker=ticker)
    # just grab last close and volume
    last = ohlcv.iloc[-1]
    price_snapshot = (
        f"{last.name.date()}: 종가={last['종가']:,}원, 거래량={int(last['거래량']):,}"
    )

    # ─── 3) Fundamental ratios ─────────────────────────────────
    fs = extract_financial_statements(
        corp=corp_code_for(ticker),  # or pass corp object if needed
        bgn_de=(date.today() - timedelta(days=365)).strftime("%Y%m%d"),
    )
    # pick most recent annual income statement
    latest_is = fs["is"].iloc[-1]
    revenue = latest_is["매출액"]
    net_income = latest_is["당기순이익"]
    fundamentals = (
        f"매출액={revenue:,}원, 당기순이익={net_income:,}원"
    )

    # ─── 4) Recent news headlines ───────────────────────────────
    news_items = get_recent_news(ticker, limit=5)  # returns list of dicts{date, title, snippet}
    news_text = "\n".join(
        f"{n['date']}: {n['title']} — {n['snippet']}" for n in news_items
    ) or "관련 뉴스 없음"

    # ─── 5) Raw “Risk Factors” from DART-FSS ───────────────────
    corp_code = corp_code_for(ticker)
    one_year_ago = (date.today() - timedelta(days=365)).strftime("%Y%m%d")
    raw_html = get_risk_factors(corp_code, bgn_de=one_year_ago, end_de=today) or ""

    # ─── 6) RAG‐powered summarization of the risk section ─────
    if raw_html:
        rag = RAGPipeline(chunk_method="section", final_k=5)
        rag.ingest(f"{ticker}_risk", raw_html)
        risk_summary = rag.answer(f"{ticker}의 위험요소 요약해줘")
    else:
        risk_summary = "위험요소 공시가 없습니다."

    # ─── 7) Final prompt composition for HyperClova ────────────
    prompt = (
        f"다음 정보를 바탕으로 {ticker}의 종합 리스크를 진단해 주세요:\n\n"
        f"1) 시장 변동성(Realized Volatility): {rv:.2%}\n"
        f"2) 최대 낙폭(Max Drawdown): {md:.2%}\n"
        f"3) 내재 변동성(Implied Volatility): {iv:.2%}\n"
        f"4) 최근 가격 스냅샷: {price_snapshot}\n"
        f"5) 주요 재무 지표: {fundamentals}\n"
        f"6) 최근 뉴스 헤드라인:\n{news_text}\n\n"
        f"7) 공시된 위험요소 요약:\n{risk_summary}\n\n"
        "이 내용을 종합해 투자자가 알아야 할 주요 위험 요인을 3~5개 포인트로 알려주세요."
    )

    # ─── 8) Call LLM for final narrative ───────────────────────
    narrative = rag.answer(prompt)

    return {
        "ticker": ticker,
        "realized_volatility": rv,
        "max_drawdown": md,
        "implied_volatility": iv,
        "price_snapshot": price_snapshot,
        "fundamentals": fundamentals,
        "recent_news": news_items,
        "risk_factors_summary": risk_summary,
        "overall_narrative": narrative,
    }
