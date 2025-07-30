
from datetime import date
from pykrx.stock import (
    get_market_ticker_list,
    get_market_cap,
    get_etf_ticker_list,   # ETF 리스트
    get_etn_ticker_list    # ETN 리스트
 )
import pandas as pd

def get_top_n_by_market_cap(
    n: int = 100,
    market: str = "ALL"
) -> list[str]:
     """
     KOSPI/KOSDAQ/ALL 시장과 ETF·ETN에서
     시가총액 상위 N개 종목(티커) 리스트를 반환

     today = date.today().strftime("%Y%m%d")
"""
    # 1) 주식 티커 (KOSPI/KOSDAQ/KONEX/ALL)
    stock_tickers = get_market_ticker_list(today, market)  # :contentReference[oaicite:0]{index=0}
    # 2) ETF·ETN 티커 추가
    etf_tickers = get_etf_ticker_list(today)               # :contentReference[oaicite:1]{index=1}
    etn_tickers = get_etn_ticker_list(today)               # :contentReference[oaicite:2]{index=2}
    all_tickers = list({*stock_tickers, *etf_tickers, *etn_tickers})
 

    # 오늘 기준 시가총액(또는 AUM) 데이터 조회
    marcap_df = get_market_cap(today)
 
     # 관심 시장에 속한 종목만 필터링

+    marcap_df = marcap_df[marcap_df['ticker'].isin(all_tickers)]
 
     # 시가총액 기준 내림차순 정렬 후 상위 N개 선택
     top_n_list = (
         marcap_df
         .nlargest(n, 'market_cap')
         ['ticker']
         .tolist()
     )
 
     return top_n_list
