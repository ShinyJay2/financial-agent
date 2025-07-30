# ㄹ) 외인*기관 최신 순매도일 수
    # 외인*기관의 매수금액 vs 매도금액만 비교
        # 순매수일 (+)
        # 순매도일 (-)
"""
    최근 30일 중 순매수합계가 음수인 일수(neg_days)에 따라
      - 0 ≤ neg_days < 10   → '낮음'
      - 10 ≤ neg_days ≤ 20  → '중간'
      -      neg_days ≥ 21  → '높음'
"""

import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pykrx.stock import get_market_trading_value_by_date
from app.risk.d_e_r import get_corp

def get_risk_components(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    리스크 계산에 필요한 일별 데이터 구성:
      - 기관합계 + 외국인합계 (순매수 금액 합)
      - 전체 매도금액 (거래대금 대용)
    반환: DataFrame with columns ['순매수합계', '거래대금']
    """
    # 1) 순매수 합계: detail=False 
    df_flow = get_market_trading_value_by_date(start, end, ticker, detail=False)
    net_flow = df_flow["기관합계"] + df_flow["외국인합계"] + df_flow["기타법인"]

    # 2) 거래대금: on='매도' → '전체' 컬럼
    df_sell = get_market_trading_value_by_date(start, end, ticker, on="매도")
    total_value = df_sell["전체"]

    # 3) 정리
    df = pd.DataFrame({
        "순매수합계": net_flow,
        "거래대금": total_value
    })
    df.index = pd.to_datetime(df.index)
    # print(df)
    return df

def get_last_30_days_dates() -> pd.DatetimeIndex:
    """
    Returns a DatetimeIndex of the last 30 calendar days (including today),
    with daily frequency.
    """
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=29)
    return pd.date_range(start=start, end=today, freq='D')


def count_negative_net_flow(df: pd.DataFrame) -> int:
    """
    Given the DataFrame from get_risk_components (indexed by date, with
    a '순매수합계' column), returns the number of days in the last
    30 days where '순매수합계' < 0.
    """
    # 1) 최근 30일 날짜 인덱스 생성
    last_30 = get_last_30_days_dates()
    # 2) 해당 날짜만 df로 재색인(reindex) → 없는 날은 NaN
    df_30 = df.reindex(last_30)
    # 3) 음수인 값만 카운트 (NaN < 0 → False 처리)
    neg_count = (df_30['순매수합계'] < 0).sum()
    return int(neg_count)

def categorize_negative_flow_days(neg_days: int) -> str:
    """
    최근 30일 중 순매수합계가 음수인 일수(neg_days)에 따라
      - 0 ≤ neg_days < 10   → '낮음'
      - 10 ≤ neg_days ≤ 20  → '중간'
      -      neg_days ≥ 21  → '높음'
    """
    if neg_days < 10:
        return "낮음"
    elif neg_days < 21:
        return "중간"
    else:
        return "높음"

# ---아래는 수정 이전 코드. 전체 대비 비율, 최장 연속 매도일수 등 계산------------------------------------
def compute_cnssd(ticker: str, start: str, end: str) -> int:
    """
    CNSSD: Consecutive Net Sell Days
      - 순매수합계/거래대금 비율이 0보다 클 때를 '순매도일'로 간주
      - 최대 연속 순매도일 수를 반환
    """
    # 1) 리스크 컴포넌트 가져오기
    df = get_risk_components(ticker, start, end)

    # 2) 일별 순매도 비율 계산
    flow_ratio = df["순매수합계"] / df["거래대금"]
    sell_ratio = flow_ratio.clip(upper=0).abs()  # 음수(순매도)만 양수로

    # 3) 순매도일(True) 마스크 생성
    mask = sell_ratio > 0

    # 4) 최장 연속 True 카운트
    max_streak = 0
    current = 0
    for is_sell_day in mask:
        if is_sell_day:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return max_streak

# 1) 누적 순매수합계 시리즈
def get_cumulative_flow(ticker: str, start: str, end: str) -> pd.Series:
    """
    순매수합계 = 기관합계 + 외국인합계
    일별로 누적합산한 시리즈 반환
    """
    df = get_market_trading_value_by_date(start, end, ticker)
    net_flow = df["기관합계"] + df["외국인합계"]+df['기타법인']
    cum_flow = net_flow.cumsum()
    cum_flow.index = pd.to_datetime(cum_flow.index)
    return cum_flow

# 2) 월별·주별 누적 추이
def get_monthly_cumulative_flow(ticker: str, start: str, end: str, months: int = 3) -> pd.Series:
    """
    최근 months개월 기준 누적 순매수 추이 (월말 기준)
    """
    cum = get_cumulative_flow(ticker, start, end)
    monthly = cum.resample('ME').last()
    return monthly.tail(months)

def get_weekly_cumulative_flow(ticker: str, start: str, end: str, weeks: int = 4) -> pd.Series:
    """
    최근 weeks주 기준 누적 순매수 추이 (주말(Fri) 기준)
    """
    cum = get_cumulative_flow(ticker, start, end)
    weekly = cum.resample('W-FRI').last()
    return weekly.tail(weeks)

# 3) 기간별 평균 리스크 (1,3,6,12개월)
def compute_periodic_average_sell_ratio(
    ticker: str,
    end: str,
    periods_months=(1, 3, 6, 12)
) -> dict:
    """
    각 기간(months)별로
      ratio_t = -순매수합계_t / 거래대금_t   (순매도→양수, 순매수→음수)
    이 ratio_t 를 **모든 일수**에 대해 평균 낸 값을 반환.
    """
    results = {}
    end_dt = datetime.strptime(end, "%Y%m%d")
    for m in periods_months:
        # 기간 시작일 계산
        start_dt = end_dt - relativedelta(months=m) + relativedelta(days=1)
        start = start_dt.strftime("%Y%m%d")
        # 해당 기간 컴포넌트 가져오기
        df = get_risk_components(ticker, start, end)
        # 순매도 비율 계산 (순매수일엔 음수 비율로 포함)
        ratio = df["순매수합계"] / df["거래대금"]
        # print("ratio:  ____ ",ratio)
        # 기간 전체 일수에 대해 평균
        results[f"{m}M"] = ratio.mean() * 100  # % 단위

    # DataFrame 생성 및 포맷팅
    result_df = pd.DataFrame.from_dict(results, orient="index", columns=["평균 순매도 비율(%)"])
    result_df.index.name = "기간"
    # 소수점 둘째 자리까지
    result_df["평균 순매도 비율(%)"] = result_df["평균 순매도 비율(%)"].map(lambda x: f"{x:.2f}%")
    return result_df["평균 순매도 비율(%)"]

# ---위에는 수정 이전 코드. 전체 대비 비율, 최장 연속 매도일수 등 계산-----------

def calculate_rank_days(ticker:str):
    corp_name, corp = get_corp(ticker)
    df = get_risk_components(ticker, "20240726", "20250726")
    # dates = get_last_30_days_dates()
    neg_days = count_negative_net_flow(df)
    level = categorize_negative_flow_days(neg_days)
    return corp, corp_name, neg_days, level

if __name__ == "__main__":
    ticker = '047050'
    corp, corp_name, neg_days, level =calculate_rank_days(ticker)
    print(f"[ticker: {ticker}], [종목명: {corp_name}] [최근 30일 중 외인*기관 순매도 일수: {neg_days}일], [위험 수준: {level}]")
    
# # ── 사용 예시 ──
# if __name__ == "__main__":
#     ticker = "247540"
#     start, end = "20240726", "20250726"

#     df = get_risk_components("247540", "20240726", "20250726")
#     # 1) 최근 30일 날짜 확인
#     dates = get_last_30_days_dates()
#     print("최근 30일 날짜:", dates.strftime("%Y-%m-%d").tolist())
    
#     # 2) 음수 순매수합계 개수
#     neg_days = count_negative_net_flow(df)
#     print(f"최근 30일 중 순매수합계가 음수인 날: {neg_days}일")
