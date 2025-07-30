# app/krx_client.py

from pykrx import stock
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_realtime_price(ticker: str) -> int:
    """
    Return the real-time quote (현재가) for a KRX ticker
    by scraping Naver Finance. Only works when the market is open.
    """
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # The current price lives in <p class="no_today"><em class="no_up">…<span class="blind">123,456</span>
    price_tag = soup.select_one("p.no_today .blind")
    if not price_tag:
        raise ValueError(f"실시간 주가를 가져올 수 없습니다: {ticker}")
    # Strip commas and convert to int
    return int(price_tag.text.strip().replace(",", ""))


LOOKBACK_DAYS = 5

def get_stock_close_price(ticker: str, target_date: str = None) -> float:
    """
    Return the most recent closing price for a Korean stock.
    1) Try PyKRX on target_date (or today), looking back up to LOOKBACK_DAYS.
    2) If no data, fallback to FinanceDataReader.
    """
    # 1) Determine the date to start lookup
    if target_date:
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        date_obj = datetime.now()

    # 2) Look back up to LOOKBACK_DAYS using PyKRX
    for _ in range(LOOKBACK_DAYS):
        day_str = date_obj.strftime("%Y%m%d")
        df = get_market_ohlcv_by_date(day_str, day_str, ticker)
        if not df.empty:
            return float(df["종가"].iloc[-1])
        date_obj -= timedelta(days=1)

    # 3) Fallback: FinanceDataReader
    try:
        # FinanceDataReader expects 'YYYY-MM-DD' format
        start = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        end   = datetime.now().strftime("%Y-%m-%d")
        df_fdr = fdr.DataReader(ticker, start, end)
        if not df_fdr.empty:
            # FDR column is 'Close'
            return float(df_fdr["Close"].iloc[-1])
    except Exception:
        pass

    raise ValueError(f"No recent price found for {ticker} in last {LOOKBACK_DAYS} days")



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



def get_index_ohlcv_by_date(
    fromdate: str,
    todate: str,
    index_name: str = "KOSPI"
) -> pd.DataFrame:
    """
    Return daily OHLCV for the given market index (KOSPI/KOSDAQ/KONEX)
    between fromdate and todate. Dates must be in YYYYMMDD format.
    """
    code = _INDEX_CODES.get(index_name.upper())
    if not code:
        raise ValueError(f"Unsupported index: {index_name!r}")
    
    fromdate_fmt = f"{fromdate[:4]}-{fromdate[4:6]}-{fromdate[6:]}"
    todate_fmt = f"{todate[:4]}-{todate[4:6]}-{todate[6:]}"
    
    df = fdr.DataReader(code, fromdate_fmt, todate_fmt)
    if df.empty:
        raise ValueError(f"No index data found for {index_name} from {fromdate} to {todate}")
    
    return df


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
