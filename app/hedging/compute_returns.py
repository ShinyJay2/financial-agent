
from pykrx.stock import (
    get_market_ohlcv_by_date
)
import pandas as pd



def fetch_price_series(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.Series:
    """
    기간(start_date~end_date)의 일별 종가 시리즈 반환
    """
    df = get_market_ohlcv_by_date(start_date, end_date, ticker)
    return df["종가"].rename(ticker)


def get_aligned_price_df(
    base_ticker: str,
    candidate_tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    기준 종목과 후보 종목들의 종가를 공통 날짜로 정렬(dropna)하여 DataFrame 반환
    """
    series_dict = {
        base_ticker: fetch_price_series(base_ticker, start_date, end_date)
    }
    for tk in candidate_tickers:
        series_dict[tk] = fetch_price_series(tk, start_date, end_date)

    price_df = pd.DataFrame(series_dict).dropna(how="any")
    return price_df


def compute_daily_returns(
    price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    종가 DataFrame을 받아 일별 수익률(returns) 계산 후 반환
    """
    return price_df.pct_change().dropna(how="any")

