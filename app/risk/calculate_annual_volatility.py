from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from pykrx import stock

# 날짜 설정
today = datetime.today()
one_year_ago = today - timedelta(days=365)
fromdate = one_year_ago.strftime("%Y%m%d")
todate = today.strftime("%Y%m%d")

# KOSPI 종목 리스트
tickers = stock.get_market_ticker_list(market="KOSPI")

# 결과 저장용 리스트
results = []

for ticker in tqdm(tickers):
    try:
        df = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
        if df.empty or "종가" not in df.columns:
            continue
        prices = df["종가"].astype(float)
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            continue
        volatility = returns.std() * np.sqrt(252)
        results.append({"ticker": ticker, "volatility": volatility})
    except Exception:
        continue

# DataFrame 변환
vol_df = pd.DataFrame(results)

# 분위수 계산
percentiles = np.percentile(vol_df["volatility"], [0, 10, 20, 40, 60, 80, 90, 100])
percentile_series = pd.Series(percentiles, index=["min", "P10", "P20", "P40", "P60", "P80", "P90", "max"])

print("\n📊 KOSPI 종목 연환산 변동성 분위수:\n")
print(percentile_series.round(4))
