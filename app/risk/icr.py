from typing import Dict, Union, Tuple
from datetime import date
from dateutil.relativedelta import relativedelta
from dart_fss.fs.fs import FinancialStatement
from typing import Dict, Optional

from dart_fss.errors.errors import NotFoundConsolidated
from dart_fss.corp import Corp, CorpList
from pandas import DataFrame
import pandas as pd

from ..utils.ticker_map import find_name_by_ticker
from app.ingestion.dart_fss_client import find_company_by_name

from app.risk.d_e_r import get_corp, compute_bgn_de

import re

def extract_financial_statements(
    corp: Corp,
    bgn_de: str,
    end_de: Optional[str] = None,
    report_tp: str = "annual",
    separate: bool = True
) -> Dict[str, DataFrame]:
    """
    Download and parse financial statements.
    - bgn_de: YYYYMMDD start date
    - end_de: YYYYMMDD end date (defaults to today)
    - report_tp: 'annual', 'half', or 'quarter'
    Returns a dict with keys 'bs','is','cis','cf'.
    """
    end_de = end_de or date.today().strftime("%Y%m%d")
    return corp.extract_fs(bgn_de=bgn_de, end_de=end_de, report_tp=report_tp, separate=separate)


def extract_cis_df(corp, bgn_de: str) -> pd.DataFrame:
    """
     포괄 손익계산서(cis) DataFrame을 꺼냄.
    연결재무제표 우선 시도, 실패 시 개별재무제표로 fallback.
    """
    # ① 연결재무제표 시도
    try:
        fs   = extract_financial_statements(corp, bgn_de=bgn_de, report_tp="annual", separate=True)
        cis_df = fs['cf']
        if cis_df is not None and not cis_df.empty:
            print("✅ 연결재무제표 사용")
            return cis_df
        else:
            print("⚠️ 연결재무제표 있음, 데이터는 비어있음 → 개별 fallback")
    except NotFoundConsolidated:
        print("⚠️ 연결재무제표 미공시 → 개별 fallback")
    except Exception as e:
        # 그 외 다른 에러
        raise

    if cis_df is None or cis_df.empty:
        raise ValueError("포괄손익계산서 데이터가 없습니다.")
    print(f"📄 포괄손익계산서 로우 수: {len(cis_df)}")
    return cis_df

def find_cols(cis_df) -> Tuple[object, object]:
    """
    account 컬럼(label_ko)과 최신 금액 컬럼(연결·개별 모두 포함)을 반환
    """
    # 모든 컬럼명 확인
    # print("🧩 모든 컬럼 목록:")
    # for col in cis_df.columns:
    #     print(col)

    # 1) account 컬럼 찾기
    account_cols = [
        col for col in cis_df.columns
        if isinstance(col, tuple) and col[1] == 'label_ko'
    ]
    if not account_cols:
        raise ValueError("label_ko 컬럼이 없습니다.")
    account_col = account_cols[0]
    # print(cis_df.columns)
    
    # 2) 금액 컬럼 (label_ko 제외 + 손익계산서 or 기간 포맷)
    amount_cols = [
        col for col in cis_df.columns
        if (
            isinstance(col, tuple)
            and isinstance(col[0], str)
            and re.match(r'\d{8}-\d{8}', col[0])
            and isinstance(col[1], tuple)
            and any(x in str(col[1]) for x in ['별도', '연결'])  # 보통 '별도재무제표' or '연결재무제표'
        )
    ]

    if not amount_cols:
        raise ValueError("금액 컬럼이 없습니다. ")
    latest_col = sorted(amount_cols)[-1]
    print(f"account_col:{account_col}, latest_col:{latest_col}")
    
        # account_col, latest_col 을 찾은 직후
    print("▶ 전체 계정명 샘플:")
    print(cis_df[account_col].drop_duplicates().tolist())

    return account_col, latest_col


def parse_icr_amounts(cis_df, account_col, latest_col) -> Tuple[float, float]:
    """
    Find Operating Income and Interest Expense in the income statement DataFrame and return them as floats.
    """
    # 1) 영업이익 (또는 손실)
    op_row = cis_df[cis_df[account_col].str.contains(r"영업.?이익|영업.?손실", regex=True)]

    # 2) 이자비용 → '금융비용'으로 대체
    interest_row = cis_df[cis_df[account_col].str.contains(r"금융.?비용", regex=True)]

    if op_row.empty or interest_row.empty:
        raise ValueError("영업이익 또는 금융비용 항목이 누락됨")

    # print("💬 전체 계정명 리스트:")
    # print(cis_df[account_col].unique().tolist())

    print("\n🔍 [영업이익 or 손실] 추출 결과:")
    print(op_row[[account_col, latest_col]])

    print("\n🔍 [이자비용] 추출 결과:")
    print(interest_row[[account_col, latest_col]])

    # 3) 금액 추출: label_ko가 아닌 latest_col에서만 뽑아야 함
    op_val = op_row[latest_col].values[0]
    int_val = interest_row[latest_col].values[0]

    print(f"🔎 op_val(raw): {op_val}")
    print(f"🔎 int_val(raw): {int_val}")


    # 실제 숫자인지 확인
    try:
        operating_income = float(str(op_val).replace(",", "").strip())
    except ValueError:
        raise ValueError(f"금액 변환 실패: op_val='{op_val}'")

    try:
        interest_expense = float(str(int_val).replace(",", "").strip())
    except ValueError:
        raise ValueError(f"금액 변환 실패: int_val='{int_val}'")


    return operating_income, interest_expense

def classify_icr(op_income: float, interest_exp: float) -> Tuple[Union[float, None], str]:
    """
    Compute the ICR and determine the risk level based on thresholds.
    """
    # Avoid division by zero
    if interest_exp != 0:
        ratio = round(op_income / interest_exp, 2)
    else:
        ratio = None  # undefined if no interest expense (could also treat as safe infinity)
    # Determine risk category
    if ratio is None:
        level = "비율 None"
    elif ratio < 3.0:
        level = "고위험"
    elif ratio < 6.0:
        level = "중간 위험"
    else:
        level = "저위험"
    return ratio, level



def calculate_icr(ticker: str) -> Dict[str, Union[str, float, None]]:
    """
    Main pipeline to calculate Interest Coverage Ratio for a given stock ticker.
    """
    try:
        corp_name, corp = get_corp(ticker)                 # Get company name and Corp object
        bgn_de = compute_bgn_de(years=1)                   # 1 year ago from today as start date
        cis_df = extract_cis_df(corp, bgn_de)                # Get income statement DataFrame (consolidated)
        account_col, latest_col = find_cols(cis_df)         # Identify the account name column and latest value column
        op_income, interest_exp = parse_icr_amounts(cis_df, account_col, latest_col)  # Get needed values
        ratio, level = classify_icr(op_income, interest_exp)  # Calculate ICR and categorize risk

        return {
            "ticker":                ticker,
            "corp_name":             corp_name,
            "operating_income":      op_income,
            "interest_expense":      interest_exp,
            "interest_coverage_ratio": ratio,
            "risk_level":            level
        }
    except ValueError as ve:
        # Handle case where consolidated data is not available
        if str(ve) == "연결재무제표 미공시":
            return {
                "ticker":     ticker,
                "corp_name":  find_name_by_ticker(ticker),
                "operating_income": None,
                "interest_expense":  None,
                "interest_coverage_ratio": None,
                "risk_level": "연결 미공시",
                "error":      str(ve)
            }
        else:
            # Other ValueErrors (e.g., missing accounts)
            return {
                "ticker": ticker,
                "corp_name": find_name_by_ticker(ticker),
                "operating_income": None,
                "interest_expense":  None,
                "interest_coverage_ratio": None,
                "risk_level": "계산실패",
                "error":      str(ve)
            }
    except Exception as e:
        # Catch-all for any other unexpected errors
        return {
            "ticker": ticker,
            "corp_name": None,
            "operating_income": None,
            "interest_expense":  None,
            "interest_coverage_ratio": None,
            "risk_level": "계산실패",
            "error": str(e)
        }

# Example usage:
if __name__ == "__main__":
    print(calculate_icr("005930"))


# 247540 에코프로비엠: 단일 포괄손익계산서
# 삼성전자: 손익계산서, 포괄손익계산서
# 포스코인터네셔널: 단일 포괄손익계산서