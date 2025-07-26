from typing import Dict, Union
from ..utils.ticker_map import find_name_by_ticker
from app.ingestion.dart_fss_client import (
    corp_code_for,
    find_company_by_name,
    extract_financial_statements
)

def calculate_d_e_ratio(ticker: str, year: str = "2024") -> Dict[str, Union[str, int, float, None]]:
    """
    Calculate Debt-to-Equity Ratio for a given stock ticker.
    
    Args:
        ticker (str): 6-digit stock ticker code (e.g., '005930')
        year (str): Business year (default = '2024')

    Returns:
        dict: Dictionary with D/E ratio and risk level
    """
    try:
        # 1. Get Corp object
        corp = find_company_by_name(find_name_by_ticker(ticker))
        corp_name = corp.corp_name
        # print("✔ corp_name:", corp_name)
        
        # 2. Extract balance sheet (annual report only)
        bgn_de = f"{year}0101"
        
        # 2. Extract balance sheet
        fs = extract_financial_statements(corp, bgn_de=bgn_de, report_tp="annual")
        bs_df = fs["bs"]

        if bs_df is None or bs_df.empty:
            raise ValueError("재무상태표 데이터가 없습니다.")
        print(f"📄 재무상태표 로우 수: {len(bs_df)}")

        # ⬇️ account 컬럼 찾기
        account_col = [col for col in bs_df.columns if isinstance(col, tuple) and col[1] == 'label_ko']
        if not account_col:
            raise ValueError("label_ko 컬럼이 없습니다.")
        account_col = account_col[0]

        # ⬇️ 최신 금액 컬럼 찾기
        amount_cols = [col for col in bs_df.columns if isinstance(col, tuple) and '연결재무제표' in str(col[1])]
        if not amount_cols:
            raise ValueError("금액 컬럼이 없습니다.")
        latest_col = sorted(amount_cols)[-1]

        # ⬇️ 항목 필터
        debt_row = bs_df[bs_df[account_col].str.contains("부채총계", na=False)]
        equity_row = bs_df[bs_df[account_col].str.contains("자본총계", na=False)]

        if debt_row.empty or equity_row.empty:
            raise ValueError("부채총계 또는 자본총계 항목이 누락됨")

        # ⬇️ 값 파싱
        debt_str = str(debt_row.iloc[0][latest_col]).replace(",", "").strip()
        equity_str = str(equity_row.iloc[0][latest_col]).replace(",", "").strip()

        debt = float(debt_str)
        equity = float(equity_str)

        # 4. Calculate ratio
        ratio = round(debt / equity, 2) if equity != 0 else None

        # 5. Risk classification
        if ratio is None:
            level = "비율 None"
        elif ratio >= 2.0:
            level = "고위험"
        elif ratio >= 1.0:
            level = "중간 위험"
        else:
            level = "저위험"

        return {
            "ticker": ticker,
            "corp_name": corp_name,
            "debt": debt,
            "equity": equity,
            "d_e_ratio": ratio,
            "risk_level": level
        }

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
    print(calculate_d_e_ratio("005930"))