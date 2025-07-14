import yfinance as yf

def get_history(ticker: str, period: str = "60d"):
    return yf.Ticker(ticker).history(period=period)

def get_implied_volatility(ticker: str) -> float:
    """
    Pull the front‐month option chain and average the implied vols.
    """
    tk = yf.Ticker(ticker)
    if not tk.options:
        return 0.0
    near = tk.options[0]
    chain = tk.option_chain(near)
    vols = []
    for df in (chain.calls, chain.puts):
        vols.extend(df["impliedVolatility"].dropna().tolist())
    return float(sum(vols) / len(vols)) if vols else 0.0
