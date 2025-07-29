# ㄷ) 부채비율(D/E; Debt-to-Equity Ratio)
# 과거 1년 D/E 구하기 수정.
    # 과거 2년 분기별 증감률 총합으로 계산

from typing import Dict, Union, Tuple, List, Optional
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from ..utils.ticker_map import find_name_by_ticker
from app.ingestion.dart_fss_client import find_company_by_name, extract_financial_statements
from dart_fss.errors.errors import NotFoundConsolidated, NoDataReceived
import calendar
from dateutil.relativedelta import relativedelta


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

def extract_bs_df(corp, bgn_de: str, end_de: Optional[str] = None) :
    """
    Corp 객체로부터 연간 재무상태표(bs) DataFrame을 추출.
    """

    try:
        fs    = extract_financial_statements(corp, bgn_de=bgn_de, end_de=end_de ,report_tp="quarter")
        # print(f"fs 전체보기: {fs}")
        bs_df = fs["bs"]
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


def classify_ratio(debt: float, equity: float) -> Tuple[Union[float, None], str]:
    """
    D/E 비율을 계산하고 위험 등급을 반환
    """
    ratio = round(debt / equity, 2) if equity != 0 else None
    if ratio is None:
        level = "비율 None"
    elif ratio >= 2.0:
        level = "고위험"
    elif ratio >= 1.0:
        level = "중간 위험"
    else:
        level = "저위험"
    return ratio, level


def compute_past_semesters(n: int = 8) -> List[str]:
    """
    최근 n개 분기의 결산일을 YYYYMMDD 형식으로 반환
    (각 분기 월의 마지막 날을 자동 계산)
    """
    ends= []
    today = date.today()
    # 분기 종료월
    semesters = [6, 12]

    # 오늘 기준으로 가장 최근 지나간 분기 구하기
    prev_sem = [m for m in semesters if m < today.month]
    if prev_sem:
        cur_m = max(prev_sem)
        cur_y = today.year
    else:
        # 1~5월인 경우, 작년 12월이 가장 최근
        cur_m = 12
        cur_y = today.year - 1

    for _ in range(n):
        # 해당 월의 말일
        last_day = calendar.monthrange(cur_y, cur_m)[1]
        ends.append(f"{cur_y:04d}{cur_m:02d}{last_day:02d}")
        # 이전 반기로 이동
        idx = semesters.index(cur_m) - 1
        if idx < 0:
            cur_m = 12
            cur_y -= 1
        else:
            cur_m = semesters[idx]
    return ends

def calculate_de_semesterly_growth(ticker: str) -> Dict[str, Union[str, List[Union[float,None]]]]:
    """
    과거 8개 분기 D/E 비율과 전분기 대비 증감률을 계산
    """
    corp_name, corp = get_corp(ticker)
    # 3sus = 반기 6개
    semester_ends = compute_past_semesters(6)

    de_ratios = []
    for s_end in semester_ends:
        
        # 1) loop 진입 확인
        print(f"▶ 처리 중인 반기말: {s_end}")
        s_date = datetime.strptime(s_end, "%Y%m%d")
        
        # 분기 3개월 → 반기는 6개월로 늘림
        start = (s_date - timedelta(days=15)).strftime("%Y%m%d")
        end   = (s_date + relativedelta(months=8)).strftime("%Y%m%d")
        print(f"   start={start}, end={end}")

        bs_df = extract_bs_df(corp=corp, bgn_de=start, end_de=end)
        if bs_df is None:
            print(f"{s_end}bs_df None 나옴!@!")
            de_ratios.append(None)
            continue
        # 여기까지 왔으면 bs_df 가 None 이 아닌 경우
        print(f"   👍 bs_df 로드 성공, 로우 수={len(bs_df)}")
        # 기존 find_cols, parse_amounts, classify_ratio 재사용
        account_col, latest_col = find_cols(bs_df)
        debt, equity = parse_amounts(bs_df, account_col, latest_col)
        print(f" date={s_end}, debt={debt}, equity={equity}")
        
        ratio, _ = classify_ratio(debt, equity)
        
        de_ratios.append(ratio)

        # 전분기 대비 증감률 구하기
    pct_changes = []
    for prev, curr in zip(de_ratios, de_ratios[1:]):
        if prev is None or curr is None:
            pct_changes.append(None)
        else:
            pct_changes.append(round((curr - prev) / prev * 100, 2))

    return {
        "ticker":            ticker,
        "corp_name":         corp_name,
        "quarter_ends":      semester_ends,
        "de_ratios":         de_ratios,
        "de_pct_changes":    pct_changes
    }

if __name__ == "__main__":
    # print(calculate_d_e_ratio("005930"))
    print(calculate_de_semesterly_growth("047050"))
