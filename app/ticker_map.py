# app/ticker_map.py

import FinanceDataReader as fdr
from functools import lru_cache

@lru_cache(maxsize=1)
def load_ticker_map() -> dict[str, str]:
    """
    Build a mapping of Korean company names to their 6-digit ticker codes
    for both KOSPI and KOSDAQ markets.
    """
    # Fetch listings for each market
    df_kospi  = fdr.StockListing("KOSPI")
    df_kosdaq = fdr.StockListing("KOSDAQ")

    # Combine and build the map: { Name: Symbol }
    combined = {**dict(zip(df_kospi["Name"],  df_kospi["Symbol"])),
                **dict(zip(df_kosdaq["Name"], df_kosdaq["Symbol"]))}
    return combined

def find_ticker(name: str) -> str | None:
    """
    Given a Korean company name (e.g. '삼성전자'), return its 6-digit symbol
    (e.g. '005930') or None if not found.
    """
    return load_ticker_map().get(name)
