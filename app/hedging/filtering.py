# 시총 상위 100개 뽑고
# 3개 지표 기준으로 중 저위험 가져오기

from datetime import date, timedelta
from pykrx.stock import get_market_ticker_list, get_market_cap, get_etf_ticker_list, get_etn_ticker_list    # ETN 리스트
import pandas as pd
# from hedge_agent_utils import get_top_n_by_market_cap
from app.risk.volatility import get_volatility_info
from app.risk.beta import get_beta_info
from app.risk.foreign_organ import calculate_rank_days
# import logging
# logging.getLogger().setLevel(logging.INFO)



def get_top_n_by_market_cap(
    n: int = 100,
    market: str = "KOSPI"
) -> list[str]:

    # 오늘 날짜 (YYYYMMDD) 포맷
    # today = date.today().strftime("%Y%m%d")
    
    # print(today)
    # 어제 날짜 (YYYYMMDD) 포맷
    today = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    
    # 해당 시장의 전체 티커 목록 조회
    # tickers = get_market_ticker_list(today)
    # print(tickers)
    
        # 1) 주식 티커 (KOSPI/KOSDAQ/KONEX/ALL)
    stock_tickers = get_market_ticker_list(today, market)  # :contentReference[oaicite:0]{index=0}
    # 2) ETF·ETN 티커 추가
    etf_tickers = get_etf_ticker_list(today)               # :contentReference[oaicite:1]{index=1}
    etn_tickers = get_etn_ticker_list(today)               # :contentReference[oaicite:2]{index=2}
    all_tickers = list({*stock_tickers, *etf_tickers, *etn_tickers})
    
    
    # 오늘 기준 시가총액 데이터 조회
    marcap_df = get_market_cap(today)


    # 인덱스를 컬럼으로 변경하고, 컬럼명 통일
    marcap_df = (
        marcap_df
        .reset_index()
        .rename(columns={
            '티커': 'ticker',
            '시가총액': 'market_cap'
        })
    )

    # 관심 시장에 속한 종목만 필터링
    # marcap_df = marcap_df[marcap_df['ticker'].isin(tickers)]
    
    marcap_df = marcap_df[marcap_df['ticker'].isin(all_tickers)]

    # 시가총액 기준 내림차순 정렬 후 상위 N개 선택
    top_n_list = (
        marcap_df
        .nlargest(n, 'market_cap')
        ['ticker']
        .tolist()
    )

    return top_n_list

def collect_risk_metrics(n: int = 100) -> pd.DataFrame:
    # 1) 시가총액 상위 N개 티커 추출
    tickers = get_top_n_by_market_cap(n, market="KOSPI")

    # 2) 결과 저장할 리스트
    records = []

    # 3) 티커별 지표 계산
    for ticker in tickers:
        try:
            # 변동성 정보 (예: {'vol': 0.25, 'level': 'MID'})
            vol_info = get_volatility_info(ticker)
            # 베타 정보 (예: {'beta': 1.1, 'level': 'LOW'})
            beta_info = get_beta_info(ticker, market="KOSPI")
            # 외인·기관 순매수일수 (예: corp, corp_name, neg_days, level)
            _corp, _corp_name, neg_days, fo_level = calculate_rank_days(ticker)

            records.append({
                "ticker": ticker,
                "volatility":       vol_info.get("volatility"),
                "vol_level":        vol_info.get("risk_level"),
                "beta":             beta_info.get("beta"),
                "beta_level":       beta_info.get("risk_level"),
                "neg_flow_days":    neg_days,
                "fo_rank_level":    fo_level,
            })
        except Exception as e:
            # 실패한 티커는 로그만 남기고 건너뜀
            # print(f"[Warning] {ticker}: {e}")
            continue

    # 4) DataFrame 생성
    df = pd.DataFrame.from_records(records)
    
        # 5) 필터링: 높음/매우 높음, 고위험, 높음 제거
    vol_bad  = ["높음", "매우 높음"]
    beta_bad = ["고위험"]
    fo_bad   = ["높음"]

    df = df[
        ~df["vol_level"].isin(vol_bad) &
        ~df["beta_level"].isin(beta_bad) &
        ~df["fo_rank_level"].isin(fo_bad)
    ].reset_index(drop=True)
    
    return df


if __name__ == '__main__':
    print(collect_risk_metrics(100))
    
