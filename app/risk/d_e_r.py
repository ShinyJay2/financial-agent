# ㄷ) 부채비율(D/E; Debt-to-Equity Ratio)
# 과거 1년 D/E 구하기 수정.
    # 과거 2년 분기별 증감률 총합으로 계산
    # D/E의 순증감

# 최근 6개 분기의 이전 분기 대비 증감률

from typing import Dict, Union, Tuple, List, Optional
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from ..utils.ticker_map import find_name_by_ticker
from app.ingestion.dart_fss_client import find_company_by_name, extract_financial_statements
from dart_fss.errors.errors import NotFoundConsolidated, NoDataReceived
import calendar


def get_corp(ticker: str) -> Tuple[str, object]:
    """
    티커 → (회사명, Corp 객체) 반환
    """
    corp_name = find_name_by_ticker(ticker)
    corp = find_company_by_name(corp_name)
    return corp_name, corp

# 

def compute_bgn_de(years: int = 2) -> str:
    """
    오늘 기준 years년 전 날짜를 YYYYMMDD 형식으로 반환
    """
    today = date.today()
    past = today - relativedelta(years=years)
    return past.strftime("%Y%m%d")


# 1) extract_bs_df 수정

def extract_bs_df(corp, bgn_de: str, end_de: Optional[str] = None, report_tp: Optional[str] = None, last_report_only: bool   = True) :
    """
    Corp 객체로부터 연간 재무상태표(bs) DataFrame을 추출.
    """

    try:
        fs    = extract_financial_statements(corp, bgn_de=bgn_de, end_de=end_de ,report_tp=report_tp, last_report_only=last_report_only)
        # print(f"fs 전체보기: {fs}")
        bs_df = fs["bs"]
        # print(f"bs_df 전체보기: {bs_df}")
        if bs_df is None:
            return None
        return bs_df

    except (NotFoundConsolidated, NoDataReceived, RuntimeError):
        # 연결이 없으면 여기로
        return None


def find_cols(bs_df) -> Tuple[object, object]:
    """
    account 컬럼(label_ko)과 최신 금액 컬럼(연결·개별 모두 포함)을 반환
    """
    # 1) account 컬럼 찾기
    account_cols = [
        col for col in bs_df.columns
        if isinstance(col, tuple) and col[1] == 'label_ko'
    ]
    if not account_cols:
        raise ValueError("label_ko 컬럼이 없습니다.")
    account_col = account_cols[0]

    # 2) 금액 컬럼 찾기: 연결재무제표 OR 개별재무제표
    amount_cols = [
        col for col in bs_df.columns
        if (
            isinstance(col, tuple)
            and (
                '연결재무제표' in str(col[1])
                or '개별재무제표' in str(col[1])
            )
        )
    ]
    if not amount_cols:
        raise ValueError("금액 컬럼이 없습니다. ('연결' 또는 '개별' 모두 없음)")
    latest_col = sorted(amount_cols)[-1]
    # print(f"account_col:{account_col}, latest_col:{latest_col}")
        # account_col, latest_col 을 찾은 직후
    # print("▶ 전체 계정명 샘플:")
    # print(bs_df[account_col].drop_duplicates().tolist())

    return account_col, latest_col


def parse_amounts(bs_df, account_col, latest_col) -> Tuple[float, float]:
    """
    bs_df에서 부채총계와 자본총계를 찾아 float로 반환
    """
    import re

    # debt 필터에 '총부채' 또는 '부채' 뒤에 '총계' 패턴 추가
    debt_row = bs_df[
        bs_df[account_col]
        .str.contains(r"(?:총부채|부채.*총계)", regex=True)
    ]
    # equity는 그대로 자본총계
    equity_row = bs_df[
        bs_df[account_col]
        .str.contains(r"자본.*총계", regex=True)
    ]

    if debt_row.empty or equity_row.empty:
        raise ValueError("부채총계 또는 자본총계 항목이 누락됨")

    debt = float(str(debt_row.iloc[0][latest_col]).replace(",", "").strip())
    equity = float(str(equity_row.iloc[0][latest_col]).replace(",", "").strip())
    return debt, equity


def classify_ratio(debt: float, equity: float) :
    """
    D/E 비율을 계산하고 위험 등급을 반환
    """
    ratio = round(debt / equity, 2) if equity != 0 else None

    return ratio

def classify_de_increment_change(total_pct_change: float) -> str:
    """
    총 부채비율 증감률(D/E_IC) 합계를 기준으로 위험 수준을 분류
    """
    if total_pct_change >= 15.0:
        return "매우 높음"
    elif 5.0 <= total_pct_change < 15.0:
        return "높음"
    elif -4.9 <= total_pct_change <= 4.9:
        return "보통"
    elif -14.9 <= total_pct_change < -5.0:
        return "낮음"
    elif total_pct_change <= -15.0:
        return "매우 낮음"
    else:
        return "분류불가"  # 예외 처리용



def compute_past_quarters(n: int = 8) -> List[str]:
    """
    최근 n개 분기의 결산일(분기 마지막 날)을 YYYYMMDD 형식으로 반환
    """
    ends = []
    today = date.today()
    quarters = [3, 6, 9, 12]

    # 가장 최근 지난 분기말 찾기
    prev_q = [m for m in quarters if m < today.month]
    if prev_q:
        cur_m = max(prev_q)
        cur_y = today.year
    else:
        cur_m = 12
        cur_y = today.year - 1

    for _ in range(n):
        last_day = calendar.monthrange(cur_y, cur_m)[1]
        ends.append(f"{cur_y:04d}{cur_m:02d}{last_day:02d}")
        # 이전 분기로 이동
        idx = quarters.index(cur_m) - 1
        if idx < 0:
            cur_m = 12
            cur_y -= 1
        else:
            cur_m = quarters[idx]
    return ends

def calculate_de_quarterly_growth(
    ticker: str
) -> Dict[str, Union[str, List[Union[float,None]]]]:
    """
    과거 `quarters`개 분기별 D/E 비율과 전분기 대비 증감률 계산
    """
    
    corp_name, corp = get_corp(ticker)
    # quarter_ends = compute_past_quarters(quarters)

    de_ratios: List[Optional[float]] = []
    
    manual_date_ranges = [
    # ("20220101", "20221231"),  # 2022년
    # ("20230101", "20231231"),  # 2023년
    # ("20240101", "20241231"),  # 2024년
    ("20250101", "20250729"),  # 2025년 ㅁ(종료일은 7월 29일)
    ]
    
    # 순서대로 뽑을 report_tp 리스트
    report_types = ["quarter", "half", "quarter", "annual"]

    de_ratios: List[Optional[float]] = []
    periods: List[str] = []

    for start, end in manual_date_ranges:
        year = start[:4]
         # 2025년만 Annual 한 번만
        if year == "2025":
            # print(f"▶ {year} annual only: {start}~{end}")
            bs_df = extract_bs_df(
                corp,
                bgn_de=start,
                end_de=end,
                report_tp="annual",
                last_report_only=True
            )
            if bs_df is None:
                de_ratios.append(None)
                # print("   → annual bs_df None!")
            else:
                account_col, latest_col = find_cols(bs_df)
                debt, equity = parse_amounts(bs_df, account_col, latest_col)
                ratio = classify_ratio(debt, equity)
                de_ratios.append(ratio)
                # print(f"   → annual D/E={ratio}")
                
            periods.append(f"{start}-{end}:annual")
            continue
        
        # 이번 연도에 대한 임시 버퍼
        year_ratios: List[Optional[float]] = []
        year_periods: List[str] = []
        for idx, rpt in enumerate(report_types):
           # 1분기(slot 0)는 placeholder, 나중에 2분기 값으로 덮어씀
            # 안덮어쓰고 그냥 제외함.
            if rpt == "quarter" and idx == 0:
                # print(f"▶ {year} 1Q: SKIP, will align to 2Q later")
                year_ratios.append(None)
                year_periods.append(f"{start}-{end}:{rpt}")
                continue
            
            #print(f"▶ {year} {rpt}: {start}~{end}")
            bs_df = extract_bs_df(corp, bgn_de=start, end_de=end, report_tp=rpt)
            if bs_df is None:
                # print(f"   → {rpt} bs_df None!")
                de_ratios.append(None)
            else:
                # print(f"   → {rpt} bs_df 로드 성공, 로우 수={len(bs_df)}")
                account_col, latest_col = find_cols(bs_df)
                debt, equity = parse_amounts(bs_df, account_col, latest_col)
                ratio = classify_ratio(debt, equity)
                de_ratios.append(ratio)
            periods.append(f"{start}-{end}:{rpt}")

    # 전분기 대비 증감률 계산
    de_pct_changes = [
        None if prev is None or curr is None else round((curr - prev) / prev * 100, 2)
        for prev, curr in zip(de_ratios, de_ratios[1:])
    ]
       # None이 아닌 값만 합산
    total_pct_change = sum(x for x in de_pct_changes if x is not None)


    return {
        "ticker":         ticker,
        "corp_name":      corp_name,
        # "quarter_ends":   quarter_ends,
        # "de_ratios":      de_ratios,
        # "de_pct_changes": de_pct_changes,
        "total_pct_change":  round(total_pct_change, 2),
        "de_ic_grade": classify_de_increment_change(total_pct_change)
    }

if __name__ == "__main__":
    print(calculate_de_quarterly_growth("047050"))


