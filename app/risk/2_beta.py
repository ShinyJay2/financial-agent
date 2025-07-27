# app/risk/beta.py

from app.ingestion.krx_client import get_market_ohlcv_by_date, get_index_ohlcv_by_date
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def classify_beta_risk(beta: float) -> str:
    if beta is None:
        return "Unknown"
    elif beta >= 1.2:
        return "고위험"
    elif beta >= 0.8:
        return "중간위험"
    else:
        return "저위험"

def get_beta_info(ticker: str, market: str = "KOSPI",
                  fromdate: str = None, todate: str = None) -> dict:
    """
    Calculate Beta of a stock against a market index (KOSPI, KOSDAQ, etc).
    Returns {ticker, beta, risk_level}
    """
    # 1. 날짜 범위 설정
    if not todate:
        todate_dt = datetime.today()
        todate = todate_dt.strftime("%Y%m%d")
    else:
        todate_dt = datetime.strptime(todate, "%Y%m%d")

    if not fromdate:
        fromdate_dt = todate_dt - timedelta(days=365)
        fromdate = fromdate_dt.strftime("%Y%m%d")

    # 2. 가격 데이터 가져오기
    try:
        df_stock = get_market_ohlcv_by_date(fromdate, todate, ticker)
        df_index = get_index_ohlcv_by_date(fromdate, todate, market)
    except Exception:
        return {
            "ticker": ticker,
            "beta": None,
            "risk_level": "Unknown"
        }

    if df_stock.empty or df_index.empty:
        return {
            "ticker": ticker,
            "beta": None,
            "risk_level": "Unknown"
        }

    # 3. 수익률 시계열 계산
    stock_returns = df_stock["종가"].astype(float).pct_change().dropna()
    index_returns = df_index["Close"].astype(float).pct_change().dropna()

    # 4. 공통 날짜로 정렬 및 정렬
    df = pd.DataFrame({"stock": stock_returns, "market": index_returns}).dropna()
    if len(df) < 2:
        return {
            "ticker": ticker,
            "beta": None,
            "risk_level": "Unknown"
        }

    # 5. 공분산 / 분산 계산
    covariance = np.cov(df["stock"], df["market"])[0][1]
    market_var = np.var(df["market"])

    if market_var == 0:
        return {
            "ticker": ticker,
            "beta": None,
            "risk_level": "Unknown"
        }

    beta = covariance / market_var

    return {
        "ticker": ticker,
        "beta": round(beta, 4),
        "risk_level": classify_beta_risk(beta)
    }

if __name__ == "__main__":
    result = get_beta_info("247540", market="KOSPI")
    print(result)
