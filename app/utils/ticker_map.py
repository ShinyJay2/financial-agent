from pykrx import stock
from functools import lru_cache
from typing import Optional, Dict, List, Tuple
import re

@lru_cache(maxsize=1)
def load_name_to_ticker() -> Dict[str, str]:
    """
    Build a mapping of Korean company names to their 6-digit tickers
    by querying KOSPI, KOSDAQ, and KONEX via PyKRX.
    """
    mapping: Dict[str, str] = {}
    for market in ("KOSPI", "KOSDAQ", "KONEX"):
        try:
            tickers = stock.get_market_ticker_list(market=market)
        except Exception:
            continue

        for t in tickers:
            try:
                name = stock.get_market_ticker_name(t)
                mapping[name] = t
            except Exception:
                continue

    return mapping

@lru_cache(maxsize=1)
def load_ticker_to_name() -> Dict[str, str]:
    """
    Reverse map of ticker → company name, built from the name → ticker map.
    """
    return {ticker: name for name, ticker in load_name_to_ticker().items()}

def find_ticker_by_name(name: str) -> Optional[str]:
    """
    Given a Korean company name (e.g. '삼성전자'),
    return its ticker (e.g. '005930') or None if not found.
    """
    return load_name_to_ticker().get(name)

def find_name_by_ticker(ticker: str) -> Optional[str]:
    """
    Given a 6-digit ticker (e.g. '005930'),
    return the Korean company name (e.g. '삼성전자') or None.
    """
    return load_ticker_to_name().get(ticker)

def all_tickers() -> List[str]:
    """Return the full list of tickers."""
    return list(load_ticker_to_name().keys())

def all_companies() -> List[str]:
    """Return the full list of company names."""
    return list(load_name_to_ticker().keys())

def ticker_name_pairs() -> List[Tuple[str, str]]:
    """Return (ticker, name) tuples for all mapped companies."""
    return list(load_ticker_to_name().items())


from functools import lru_cache
from typing import Optional, Dict, List, Tuple
from pykrx import stock

# --- your existing name↔ticker loaders here (load_name_to_ticker, etc.) ---

def extract_tickers_from_query(query: str) -> List[str]:
    """
    Greedy, longest-match extraction of tickers from Korean company names in the query.
    E.g. "에코프로비엠 위험성" → ["247540"] only, not ["086520","247540"].
    """
    name_to_ticker = load_name_to_ticker()
    # Sort names so that longer ones match before shorter substrings
    sorted_names = sorted(name_to_ticker.keys(), key=len, reverse=True)

    found = []
    q = query  # we'll blank out matched spans as we go
    for name in sorted_names:
        idx = q.find(name)
        if idx != -1:
            found.append(name_to_ticker[name])
            # replace the matched slice with spaces so we don't rematch substrings
            q = q[:idx] + " " * len(name) + q[idx+len(name):]

    return found

