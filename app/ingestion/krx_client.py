# app/krx_client.py

from pykrx import stock
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import pandas as pd
import numpy as np

LOOKBACK_DAYS = 5

def get_stock_close_price(ticker: str, target_date: str = None) -> float:
    """
    Return the most recent closing price for a Korean stock (KOSPI/KOSDAQ/KONEX).
    1) Try PyKRX on target_date (or today) and look back up to LOOKBACK_DAYS.
    2) If no data, fallback to FDR full history.
    """
    # determine base date
    date_obj = datetime.today() if not target_date else datetime.strptime(target_date, "%Y%m%d")

    # 1) PyKRX lookback
    for i in range(LOOKBACK_DAYS):
        day = (date_obj - timedelta(days=i)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv(day, day, ticker)
        if not df.empty:
            return float(df["종가"].iloc[-1])

    # 2) fallback
    return _get_fdr_last_close(ticker)


def _get_fdr_last_close(ticker: str) -> float:
    """
    Fallback via FinanceDataReader full history → last available Close.
    """
    df = fdr.DataReader(ticker)  # covers from 2000-01-01 to today
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    return float(df["Close"].iloc[-1])


# map friendly index names to FDR codes
_INDEX_CODES = {
    "KOSPI":  "KS11",
    "KOSDAQ": "KQ11",
    "KONEX":  "KX11",  # if supported; otherwise omit
}

def get_index_close(index_name: str = "KOSPI") -> float:
    """
    Return the latest closing value for a market index via FDR.
    index_name must be one of the keys in _INDEX_CODES.
    """
    code = _INDEX_CODES.get(index_name.upper())
    if not code:
        raise ValueError(f"Unsupported index: {index_name!r}")

    df = fdr.DataReader(code)
    if df.empty:
        raise ValueError(f"No index data found for {index_name}")
    return float(df["Close"].iloc[-1])


def list_tickers(market: str = "KOSPI") -> list[str]:
    """
    Return all 6-digit tickers for KOSPI/KOSDAQ/KONEX via PyKRX.
    """
    return stock.get_market_ticker_list(market=market)


def get_realtime_price(ticker: str) -> float:
    """
    Alias for get_stock_close_price for symmetry.
    """
    return get_stock_close_price(ticker)


def get_market_ohlcv_by_date(
    fromdate: str,
    todate: str,
    ticker: str
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for `ticker` between `fromdate` and `todate` (YYYYMMDD).
    """
    return stock.get_market_ohlcv(fromdate, todate, ticker)


# app/risk/volatility.py 에서 다룰 것임

# def get_historical_volatility(
#     ticker: str, period_days: int = 252, window: int = 30
# ) -> float:
#     """
#     Fetch the last `period_days` of daily close prices,
#     compute rolling std. dev. over `window` days, and return the latest.
#     """
#     df = get_market_ohlcv_by_date(fromdate="20220101", todate="20251231", ticker=ticker)
#     # assume df has '종가' column
#     prices = df["종가"].astype(float)
#     returns = prices.pct_change().dropna()
#     vol = returns.rolling(window).std() * np.sqrt(252)
#     return float(vol.iloc[-1])
