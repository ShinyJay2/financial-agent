import yfinance as yf

def get_history(ticker: str, period: str = "60d"):
    return yf.Ticker(ticker).history(period=period)