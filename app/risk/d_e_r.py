from typing import Dict, Union, Tuple
from datetime import date
from dateutil.relativedelta import relativedelta

from ..utils.ticker_map import find_name_by_ticker
from app.ingestion.dart_fss_client import find_company_by_name, extract_financial_statements


def get_corp(ticker: str) -> Tuple[str, object]:
    """
    티커 → (회사명, Corp 객체) 반환
    """
    corp_name = find_name_by_ticker(ticker)
    corp = find_company_by_name(corp_name)
    return corp_name, corp


def compute_bgn_de(years: int = 1) -> str:
    """
    오늘 기준 years년 전 날짜를 YYYYMMDD 형식으로 반환
    """
    today = date.today()
    past = today - relativedelta(years=years)
    return past.strftime("%Y%m%d")


# 1) extract_bs_df 수정

from dart_fss.errors.errors import NotFoundConsolidated

def extract_bs_df(corp, bgn_de: str) -> 'pd.DataFrame':
    """
    Corp 객체로부터 연간 재무상태표(bs) DataFrame을 추출.
    연결재무제표가 없으면 명시적 에러로 전환.
    """
    try:
        fs    = extract_financial_statements(corp, bgn_de=bgn_de, report_tp="annual")
        bs_df = fs["bs"]
    except NotFoundConsolidated:
        # 연결이 없으면 여기로
        raise ValueError("연결재무제표 미공시")
    except Exception as e:
        # 그 외 다른 에러
        raise

    if bs_df is None or bs_df.empty:
        raise ValueError("재무상태표 데이터가 없습니다.")
    print(f"📄 재무상태표 로우 수: {len(bs_df)}")
    return bs_df



# def extract_bs_df(corp, bgn_de: str) -> 'pd.DataFrame':
#     """
#     Corp 객체로부터 연간 재무상태표(bs) DataFrame을 추출
#     """
#     fs = extract_financial_statements(corp, bgn_de=bgn_de, report_tp="annual")
#     bs_df = fs["bs"]
#     if bs_df is None or bs_df.empty:
#         raise ValueError("재무상태표 데이터가 없습니다.")
#     print(f"📄 재무상태표 로우 수: {len(bs_df)}")
#     return bs_df


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
    print("▶ 전체 계정명 샘플:")
    print(bs_df[account_col].drop_duplicates().tolist())

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


def calculate_d_e_ratio(ticker: str) -> Dict[str, Union[str, int, float, None]]:
    """
    종목 티커에 대해 D/E 비율 계산 파이프라인
    """
    try:
        corp_name, corp = get_corp(ticker)
        bgn_de = compute_bgn_de(years=1)
        bs_df = extract_bs_df(corp, bgn_de) # 여기서 ValueError("연결재무제표 미공시") 발생 가능
        account_col, latest_col = find_cols(bs_df)
        debt, equity = parse_amounts(bs_df, account_col, latest_col)
        ratio, level = classify_ratio(debt, equity)

        return {
            "ticker": ticker,
            "corp_name": corp_name,
            "debt": debt,
            "equity": equity,
            "d_e_ratio": ratio,
            "risk_level": level
        }
    except ValueError as ve:
    # 연결재무제표 미공시일 때만 특별 처리
        if str(ve) == "연결재무제표 미공시":
            return {
                "ticker":     ticker,
                "corp_name":  find_name_by_ticker(ticker),
                "debt":       None,
                "equity":     None,
                "d_e_ratio":  None,
                "risk_level": "연결 미공시",
                "error":      str(ve)
            }
    # 그 외 ValueError는 기존 처리

    except Exception as e:
        return {
            "ticker": ticker,
            "corp_name": None,
            "debt": None,
            "equity": None,
            "d_e_ratio": None,
            "risk_level": "계산실패",
            "error": str(e)
        }

if __name__ == "__main__":
    print(calculate_d_e_ratio("247540"))
