# app/risk/volatility.py

from app.ingestion.krx_client import get_market_ohlcv_by_date
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def classify_risk(vol: float) -> str:
    """
    Classify annualized volatility into risk categories.
    """
    if vol >= 0.70:
        return "매우 높음"    # 상위 10%
    elif vol >= 0.58:
        return "높음"        # P80 이상
    elif vol >= 0.35:
        return "중간"        # P40 이상
    elif vol >= 0.21:
        return "낮음"       # P10 이상
    else:
        return "매우 낮음"

def get_volatility_info(
    ticker: str,
    fromdate: str = None,
    todate: str = None
) -> dict:
    """
    Calculate 1Y annualized volatility for a given ticker using daily returns.
    Returns volatility (float) and risk_level (str).
    """
    # 기본값: 오늘 기준 최근 1년
    if not todate:
        todate_dt = datetime.today()
        print(todate_dt)
        todate = todate_dt.strftime("%Y%m%d")
    else:
        todate_dt = datetime.strptime(todate, "%Y%m%d")

    if not fromdate:
        fromdate_dt = todate_dt - timedelta(days=365)
        fromdate = fromdate_dt.strftime("%Y%m%d")

    # 1) OHLCV 데이터 조회
    df = get_market_ohlcv_by_date(fromdate, todate, ticker)

    if df.empty or "종가" not in df.columns:
        return {
            "ticker": ticker,
            "volatility": None,
            "risk_level": "Unknown"
        }

    # 2) 일일 수익률 계산
    prices = df["종가"].astype(float)
    returns = prices.pct_change().dropna()

    if len(returns) < 2:
        # 꼭 필요하진 않지만 "데이터가 극히 부족할 때도 안전하게 동작하게 하려면 유용"
            # 특히 테스트 환경, 신규 상장 종목, 휴장일 많은 종목 등에선 returns 길이가 1이 될 수 있음
        return {
            "ticker": ticker,
            "volatility": None,
            "risk_level": "Unknown"
        }

    # 3) 연환산 표준편차 계산
    volatility = returns.std() * np.sqrt(252)

    return {
        "ticker": ticker,
        "volatility": round(volatility, 4),
        "risk_level": classify_risk(volatility)
    }

if __name__ == "__main__":
    result = get_volatility_info("247540")
    print(result)

# 출력 예시:
# {'ticker': '005930', 'volatility': 0.1423, 'risk_level': 'Low'}
qqq