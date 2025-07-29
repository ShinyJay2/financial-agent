# 이 주식을 외인이 순매도
    # 외인의 매수금액 vs 매도금액만 비교
        # 순매수일 (+)
        # 순매도일 (-)

# 0~10
# 10~20
# 20~30

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pykrx.stock import get_market_trading_value_by_date

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
    return df

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

# def daily_risk_score(ticker: str, start: str, end: str) -> pd.Series:
#     """
#     일별 리스크 스코어 계산:
#       risk = max(0, -순매수합계 / 거래대금)
#     순매도일에만 비율이 양수, 순매수일은 0
#     반환: DatetimeIndex → risk_score (0~1)
#     """
#     df = get_risk_components(ticker, start, end)
#     net_sell = df["순매수합계"].clip(upper=0).abs()
#     risk = net_sell / df["거래대금"]
#     return risk.round(4)

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

# ── 사용 예시 ──
if __name__ == "__main__":
    ticker = "247540"
    start, end = "20240726", "20250726"

    # print(get_cumulative_flow(ticker, start, end))

    # 1) 누적 순매수곡선
    print("월별 누적 순매수 (최근 3개월):")
    print(get_monthly_cumulative_flow(ticker, start, end))
    print("\n주별 누적 순매수 (최근 4주):")
    print(get_weekly_cumulative_flow(ticker, start, end))

    # 2) CNSSD 건드리지 않음(이미 compute_cnssd() 있음)

    # 3) 평균 리스크 (1,3,6,12M)
    print("\n기간별 평균 리스크:")
    print(compute_periodic_average_sell_ratio(ticker, end))

    cnssd = compute_cnssd(ticker, start, end)
    print(f"CNSSD (최대 연속 순매도일수): {cnssd}일") 