COMPANY_TICKERS: dict[str,str] = {
    "삼성전자": "005930.KS",
    "현대차":   "005380.KS",
    "LG에너지솔루션": "373220.KS",
    # …add your top N names here…
}

def find_ticker(name: str) -> str | None:
    return COMPANY_TICKERS.get(name)
