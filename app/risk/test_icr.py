from dart_fss.errors.errors import NotFoundConsolidated
from dart_fss.fs.fs import FinancialStatement
import pandas as pd
from app.ingestion.dart_fss_client import find_company_by_name, extract_financial_statements

from typing import Dict, Union, Tuple
from datetime import date
from dateutil.relativedelta import relativedelta

import re
from dart_fss.errors.errors import NotFoundConsolidated
from pandas import DataFrame
import pandas as pd

from ..utils.ticker_map import find_name_by_ticker
from app.ingestion.dart_fss_client import find_company_by_name, extract_financial_statements

from app.risk.d_e_r import get_corp, compute_bgn_de


from dart_fss.errors.errors import NotFoundConsolidated
from dart_fss.fs.fs import FinancialStatement
import pandas as pd

def extract_is_df(corp, bgn_de: str) -> pd.DataFrame:
    """
    Corp 객체로부터 연간 재무상태표(bs) DataFrame을 추출.
    에러가 나면 내부 구조를 출력해 줍니다.
    """
    try:
        # 1) 원본 fs 얻기
        fs = extract_financial_statements(corp, bgn_de=bgn_de, report_tp="annual")

        # 2) fs 타입과 속성 찍어 보기
        print("🚧 extract_financial_statements() 리턴 타입:", type(fs))
        if isinstance(fs, FinancialStatement):
            # FinancialStatement 라면 to_dict() 해서 살펴본다
            d = fs.to_dict()
            print("🚧 FinancialStatement.to_dict() keys:", list(d.keys()))
            # 각 키마다 type 과 empty 여부 확인
            for k, v in d.items():
                print(f"    - {k!r}: {type(v)}, empty? {v is None or (hasattr(v,'empty') and v.empty)}")
        else:
            # dict 로 바로 왔을 때
            print("🚧 fs.keys():", list(fs.keys()))
            for k, v in fs.items():
                print(f"    - {k!r}: {type(v)}, empty? {v is None or (hasattr(v,'empty') and v.empty)}")

        # 3) 진짜 bs_df 추출
        if isinstance(fs, FinancialStatement):
            bs_df = fs.to_dict().get("bs")
        else:
            bs_df = fs.get("bs")

    except NotFoundConsolidated:
        raise ValueError("연결재무제표 미공시")
    except Exception:
        # 내부 구조 살펴보느라 발생한 에러를 그대로 올려줌
        raise

    if bs_df is None or bs_df.empty:
        raise ValueError("재무상태표 데이터가 없습니다.")
    print(f"📄 재무상태표 로우 수: {len(bs_df)}")
    return bs_df



def parse_icr_amounts(is_df, account_col, latest_col) -> Tuple[float, float]:
    """
    Find Operating Income and Interest Expense in the income statement DataFrame and return them as floats.
    """
    # Search for Operating Income row (영업이익 or 영업손실)
    op_row = is_df[ is_df[account_col].str.contains(r"(?:영업\s*이익|영업\s*손실)", regex=True) ]
    # Search for Interest Expense row (이자비용)
    interest_row = is_df[ is_df[account_col].str.contains(r"이자\s*비용", regex=True) ]

    if op_row.empty or interest_row.empty:
        raise ValueError("영업이익 또는 이자비용 항목이 누락됨")

    # Extract values (remove commas and whitespace, then convert to float)
    op_str = str(op_row.iloc[0][latest_col]).replace(",", "").strip()
    int_str = str(interest_row.iloc[0][latest_col]).replace(",", "").strip()
    operating_income = float(op_str)
    interest_expense = float(int_str)
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


def find_cols(bs_df) -> Tuple[object, object]:
    # 1) 다중인덱스(label_ko, 연결재무제표 등) 를 쓰는 경우
    account_cols = [col for col in bs_df.columns if isinstance(col, tuple) and col[1] == 'label_ko']
    amount_cols  = [col for col in bs_df.columns
                    if isinstance(col, tuple) and ('연결재무제표' in str(col[1]) or '개별재무제표' in str(col[1]))]
    if account_cols and amount_cols:
        account_col = account_cols[0]
        latest_col  = sorted(amount_cols)[-1]
    else:
        # 2) 단일 컬럼(account_nm, thstrm_amount 등) 을 쓰는 경우
        if 'account_nm' in bs_df.columns and 'thstrm_amount' in bs_df.columns:
            account_col = 'account_nm'
            latest_col  = 'thstrm_amount'
        else:
            raise ValueError(f"계정명/금액 컬럼을 찾을 수 없습니다. columns={list(bs_df.columns)}")

     # account_col, latest_col 을 찾은 직후
    print(f"▶ account_col = {account_col!r}, latest_col = {latest_col!r}")
    print("▶ 전체 계정명 샘플:", bs_df[account_col].drop_duplicates().tolist())
    return account_col, latest_col

def calculate_icr(ticker: str) -> Dict[str, Union[str, float, None]]:
    """
    Main pipeline to calculate Interest Coverage Ratio for a given stock ticker.
    """
    try:
        corp_name, corp = get_corp(ticker)                 # Get company name and Corp object
        bgn_de = compute_bgn_de(years=1)                   # 1 year ago from today as start date
        is_df = extract_is_df(corp, bgn_de)                # Get income statement DataFrame (consolidated)
        account_col, latest_col = find_cols(is_df)         # Identify the account name column and latest value column
        op_income, interest_exp = parse_icr_amounts(is_df, account_col, latest_col)  # Get needed values
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
    print(calculate_icr("047050"))


# 247540 에코프로비엠: 단일 포괄손익계산서
# 삼성전자: 손익계산서, 포괄손익계산서
# 포스코인터네셔널: 단일 포괄손익계산서