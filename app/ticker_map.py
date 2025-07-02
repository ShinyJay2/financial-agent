# app/ticker_map.py

from pykrx import stock
from functools import lru_cache
from typing import Optional, Dict

@lru_cache(maxsize=1)
def load_ticker_map() -> Dict[str, str]:
    """
    Build a mapping of Korean company names to their 6-digit tickers
    by querying KOSPI, KOSDAQ, and KONEX via PyKRX.
    """
    mapping: Dict[str, str] = {}
    for market in ("KOSPI", "KOSDAQ", "KONEX"):
        # get_market_ticker_list returns e.g. ["005930", "000660", …]
        tickers = stock.get_market_ticker_list(market=market)
        for t in tickers:
            try:
                # get_market_ticker_name("005930") → "삼성전자"
                name = stock.get_market_ticker_name(t)
                mapping[name] = t
            except Exception:
                # skip any tickers that fail to resolve
                continue
    return mapping

def find_ticker(name: str) -> Optional[str]:
    """
    Given a Korean company name (e.g. '삼성전자'),
    return its ticker (e.g. '005930') or None if not found.
    """
    return load_ticker_map().get(name)
