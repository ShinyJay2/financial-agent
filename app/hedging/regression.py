

from datetime import date, timedelta
import pandas as pd
import numpy as np

from app.hedging.filtering import collect_risk_metrics
from app.hedging.compute_returns import get_aligned_price_df, compute_daily_returns
from app.utils.ticker_map import find_name_by_ticker
from sklearn.linear_model import LinearRegression


def compute_regression(
    base_returns: pd.Series,
    target_returns: pd.Series
) -> tuple[float, float, float]:
    """
    base_returns를 설명변수 X, target_returns를 종속변수 y로
    단순선형회귀를 수행하여 (slope, intercept, r²)를 반환합니다.
    """
    X = base_returns.values.reshape(-1, 1)
    y = target_returns.values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    return slope, intercept, r2

def compute_candidate_regressions(
    returns_df: pd.DataFrame,
    base_ticker: str
) -> pd.DataFrame:
    """
    returns_df의 base_ticker 컬럼 대비 나머지 컬럼들에 대해
    회귀분석을 수행하고, slope·intercept·r2·correlation을 정리합니다.
    """
    results = []
    base = returns_df[base_ticker]
    for tk in returns_df.columns.drop(base_ticker):
        slope, intercept, r2 = compute_regression(base, returns_df[tk])
        corr = base.corr(returns_df[tk])
        results.append({
            "ticker":    tk,
            "slope":     slope, #회귀계수
            "intercept": intercept, #절편
            "r2":        r2, #결정계수
            "corr":      corr #피어슨 상관계수
        })
    return pd.DataFrame(results).sort_values("corr")

def run_hedge_pipeline(
    base_ticker: str,
    n_candidates: int = 100,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    1) collect_risk_metrics로 리스크 필터링된 상위 n_candidates 종목을 뽑고
    2) base_ticker와 이들 후보의 일별 수익률 회귀·상관분석을 수행한 뒤
    3) corr < 0인 종목만 남겨서 반환합니다.

    Returns:
        pd.DataFrame: ['ticker','slope','intercept','r2','corr'] 컬럼, corr 오름차순 정렬
    """
    # 1) 리스크 필터링
    df_metrics = collect_risk_metrics(n_candidates)
    # print(f"[Debug] 후보 리스크 통과 종목 수: {len(df_metrics)}") 
    candidates = df_metrics["ticker"].tolist()

    # 2) 기간 계산
    end       = date.today()
    start     = end - timedelta(days=lookback_days)
    start_str = start.strftime("%Y%m%d")
    end_str   = end.strftime("%Y%m%d")

    # 3) 가격 정렬 및 수익률 계산
    price_df   = get_aligned_price_df(base_ticker, candidates, start_str, end_str)
    returns_df = compute_daily_returns(price_df)
    # print(f"[Debug] returns_df shape: {returns_df.shape}")         # ← 여기

    # 4) 회귀·상관분석
    df_reg = compute_candidate_regressions(returns_df, base_ticker)

    # 5) 음(–)상관 필터링
    df_neg = df_reg[df_reg["corr"] < 0].reset_index(drop=True)
    # print(f"[Debug] 음상관 종목 수: {len(df_neg)}")  
    return df_neg


if __name__ == "__main__":
    result = run_hedge_pipeline(base_ticker="247540", n_candidates=100, lookback_days=365)
    print(result) #이거 주석처리해도됨
    ticker_list = result['ticker'].tolist()
    name_list = []
    for tic in ticker_list:
        name = find_name_by_ticker(tic)
        name_list.append(name)
    print(name_list)


