# ㅁ) 순이익 변동성(Earnings Volatility)
    # 수익률 변동성의 회계버전 

# ㄱ)조건적용

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
from dart_fss.corp import Corp
from app.risk.d_e_r import get_corp, compute_bgn_de
from pandas import DataFrame
from datetime import date
from ..utils.ticker_map import find_name_by_ticker
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
    Returns a dict with keys 'bs','cis','cf', 'is'
    ‘bs’ 재무상태표, ‘is’ 손익계산서, ‘cis’ 포괄손익계산서, ‘cf’ 현금흐름표
    """
    end_de = end_de or date.today().strftime("%Y%m%d")
    return corp.extract_fs(bgn_de=bgn_de, end_de=end_de, report_tp=report_tp, separate=separate)


def extract_cis_df(corp, bgn_de: str) -> pd.DataFrame:
    """
     (포괄) 손익계산서(cis) DataFrame을 꺼내고 없으면 손익계산서(is)를 꺼냄
    """
    # 재무제표 전체 추출 시도
    fs   = extract_financial_statements(corp, bgn_de=bgn_de, report_tp="annual", separate=True)
    
    cis_flag = True
    cis_df = fs["cis"]
    if cis_df is None or cis_df.empty:
        print("CIS 데이터 없음 → IS로 폴백")
        cis_flag = False
        cis_df = fs['is']

    # 그래도 없으면 에러
    if cis_df is None or cis_df.empty:
        raise ValueError("포괄손익계산서(cis) 및 손익계산서(is) 데이터가 모두 없습니다.")

    return cis_df


def find_net_income_label(cis_df) -> Tuple[Tuple, str]:
    """
    account 컬럼(label_ko)과 최신 금액 컬럼(연결·개별 모두 포함)을 반환
    후 당기순이익 추출
    """

    # account_col 찾기
    account_cols = [
        col for col in cis_df.columns
        if isinstance(col, tuple) and col[1] == 'label_ko'
    ]
    if not account_cols:
        raise ValueError("label_ko 컬럼이 없습니다.")
    account_col = account_cols[0]

    # net_label 분기
    labels = cis_df[account_col].unique().tolist()
    pattern = re.compile(r"^당기순이익(?:\(손실\))?$")

    net_label = next((lbl for lbl in labels if pattern.match(lbl)), None)
    if not net_label:
        raise ValueError("당기순이익 계정명이 없습니다.")

    return account_col, net_label

# 3) 금액(기간) 컬럼만 추출
def find_amount_cols(cis_df: DataFrame) -> List[Tuple]:
    """
    cis_df에서 'YYYYMMDD-YYYYMMDD' 패턴의 금액 컬럼 리스트 반환
    """
    amount_cols = [
        col for col in cis_df.columns
        if (
            isinstance(col, tuple)
            and isinstance(col[0], str)
            and re.match(r'\d{8}-\d{8}', col[0])
            and isinstance(col[1], tuple)
            and any(x in str(col[1]) for x in ['별도', '연결'])
        )
    ]
    if not amount_cols:
        raise ValueError("금액(기간) 컬럼이 없습니다.")
    return sorted(amount_cols)

# 4) 당기순이익 시계열 Series 생성
def extract_net_income_series(
    cis_df: DataFrame,
    account_col: Tuple,
    net_label: str,
    amount_cols: List[Tuple]
) -> pd.Series:
    """
    지정된 account_col, net_label, amount_cols를 이용해
    시계열(pd.Series)로 반환 (float, NaN 제거)
    """
    row = cis_df[cis_df[account_col] == net_label]
    if row.empty:
        raise ValueError(f"{net_label} 행이 없습니다.")
    raw = row[amount_cols].iloc[0].tolist()
    series = pd.Series(raw).apply(
        lambda x: float(str(x).replace(',', '')) if pd.notnull(x) else np.nan
    ).dropna()

    return series

# 5) EVI 계산
def compute_evi(series: pd.Series) -> float:
    """
    EVI = 표준편차(std, ddof=1) ÷ |평균(mean)|
    평균이 0이면 nan 반환
    """
    mean_ = series.mean()
    std_  = series.std(ddof=1)
    if mean_ == 0:
        return float('nan')

    return float(std_ / abs(mean_))

def classify_evi(evi: float) -> str:
    """
    EVI 기준에 따른 리스크 레벨 분류
    - evi < 0.5: Low
    - 0.5 ≤ evi < 1.25: Medium
    - evi ≥ 1.25: High
    """
 
    if pd.isna(evi):
        return "Unknown"
    if evi < 0.5:
        return "Low"
    elif evi < 1.25:
        return "Medium"
    else:
        return "High"

def calculate_evi(
    ticker: str,
    years: int = 3
) -> Dict[str, Union[str, float]]:
    """
    주어진 ticker에 대해 EVI를 한 번에 계산하여
    {'ticker','corp_name','net_label','evi'}를 반환합니다.
    """
    # 1) Corp 객체 가져오기
    corp_name, corp = get_corp(ticker)
    # 2) 조회 시작일 계산
    bgn_de = compute_bgn_de(years=years)
    # 3) cis DataFrame 추출
    cis_df = extract_cis_df(corp, bgn_de)
    # 4) account_col, net_label 결정
    account_col, net_label = find_net_income_label(cis_df)
    # 5) 기간별 금액 컬럼 리스트
    amount_cols = find_amount_cols(cis_df)
    # 6) 당기순이익 시계열 Series
    series = extract_net_income_series(cis_df, account_col, net_label, amount_cols)
    # 7) EVI 계산
    evi_value = compute_evi(series)
    # 8) 등급계산
    rank = classify_evi(evi_value)
  
    return {
        "ticker":      ticker,
        "corp_name":   corp_name,
        # "net_label":   net_label,
        "evi":         round(evi_value, 4),
        "rank": rank
    }

# ──────────────────────────────────────────────────────────────────────────────
# __main__ 간소화 예시
if __name__ == "__main__":
    result = calculate_evi("023530", years=3)
    print(result)
    # 출력 예시: {'ticker': '005930', 'corp_name': '삼성전자', 'net_label': '당기순이익', 'evi': 0.1234}

# # ──────────────────────────────────────────────────────────────────────────────
# # 실행 예시 (__main__)
# if __name__ == "__main__":
#     # 1) Corp 가져오기
#     ticker = "330590"
#     corp_name, corp = get_corp(ticker)

#     # 2) 조회 시작일 계산 (연 단위)
#     bgn_de = compute_bgn_de(years=3)

#     # 3) CIS DF
#     cis_df = extract_cis_df(corp, bgn_de)
#     # print(f"CIS rows: {len(cis_df)}")

#     # 4) account_col & net_label
#     account_col, net_label = find_net_income_label(cis_df)
#     print("▶ net_label:", net_label)

#     # 5) amount_cols
#     amount_cols = find_amount_cols(cis_df)
#     print("▶ amount_cols:", amount_cols)

#     # 6) 시계열 추출
#     series = extract_net_income_series(cis_df, account_col, net_label, amount_cols)
#     print("▶ Net Income Series:")
#     print(series)

#     # 7) EVI 계산
#     evi_value = compute_evi(series)
#     print(f"{corp_name} ({ticker}) EVI: {evi_value:.4f}")