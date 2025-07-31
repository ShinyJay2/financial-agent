
from datetime import date, timedelta
import numpy as np

from app.hedging.compute_returns import get_aligned_price_df, compute_daily_returns
from app.hedging.regression import run_hedge_pipeline
from app.utils.ticker_map import find_name_by_ticker

def diversification_effect(
    base_ticker: str,       # 예: '247540'
    hedge_ticker: str,      # 예: '003230'
    weight_base: float,     # 예: 0.5
    weight_hedge: float,    # 예: 0.5
    start_date: str,        # 'YYYYMMDD'
    end_date: str           # 'YYYYMMDD'
) -> tuple[str, dict]:
    """
    두 종목 포트폴리오(가중치) VaR/ES 및 단일 대비 감소율 계산.
    반환값: (hedge_ticker, {VaR_base, VaR_port, VaR_reduction_%, ES_base, ES_port, ES_reduction_%})
    """
    # A) 수익률 계산
    price_df   = get_aligned_price_df(base_ticker, [hedge_ticker], start_date, end_date)
    returns_df = compute_daily_returns(price_df)

    # B) 단일 VaR·ES (base only)
    base_ret  = returns_df[base_ticker]
    var_base  = -np.percentile(base_ret, 5)
    es_base   = -base_ret[base_ret <= np.percentile(base_ret, 5)].mean()

    # C) 포트폴리오 VaR·ES
    w         = np.array([weight_base, weight_hedge])
    port_ret  = returns_df[[base_ticker, hedge_ticker]].dot(w)
    var_port  = -np.percentile(port_ret, 5)
    es_port   = -port_ret[port_ret <= np.percentile(port_ret, 5)].mean()

    # D) 리스크 감소율
    var_red   = (var_base  - var_port) / var_base * 100
    es_red    = (es_base   - es_port ) / es_base  * 100

    metrics = {
        'VaR_base':        round(float(var_base), 2),
        'VaR_port':        round(float(var_port), 2),
        'VaR_reduction_%': round(float(var_red), 2),
        # 'ES_base':         float(es_base),
        # 'ES_port':         float(es_port),
        # 'ES_reduction_%':  float(es_red)
    }

    return hedge_ticker, metrics


def diversification_new(
    base_ticker: str,       # 예: '247540'
    hedge_ticker: str,      # 예: '003230'
    weight_base: float,     # raw relative size for base
    weight_hedge: float,    # raw relative size for hedge
    start_date: str,        # 'YYYYMMDD'
    end_date: str           # 'YYYYMMDD'
) -> tuple[str, dict]:
    """
    두 종목 포트폴리오(raw weights) VaR/ES 및 단일 대비 감소율 계산.
    weight_base/weight_hedge를 합이 1이 되도록 정규화하여 포트폴리오 비중을 계산합니다.
    반환값: (hedge_ticker, {VaR_base, VaR_port, VaR_reduction_%, ES_base, ES_port, ES_reduction_%})
    """
    # A) 수익률 계산
    price_df   = get_aligned_price_df(base_ticker, [hedge_ticker], start_date, end_date)
    returns_df = compute_daily_returns(price_df)

    # B) 단일 VaR·ES (base only)
    base_ret  = returns_df[base_ticker]
    var_base  = -np.percentile(base_ret, 5)
    es_base   = -base_ret[base_ret <= np.percentile(base_ret, 5)].mean()

    # C) 포트폴리오 VaR·ES
    w_raw     = np.array([weight_base, weight_hedge], dtype=float)
    w         = w_raw / w_raw.sum()  # normalize to fractions
    port_ret  = returns_df[[base_ticker, hedge_ticker]].dot(w)
    var_port  = -np.percentile(port_ret, 5)
    es_port   = -port_ret[port_ret <= np.percentile(port_ret, 5)].mean()

    # D) 리스크 감소율
    var_red   = (var_base  - var_port) / var_base * 100
    es_red    = (es_base   - es_port ) / es_base  * 100

    metrics = {
        'VaR_base':        round(float(var_base), 2),
        'VaR_port':        round(float(var_port), 2),
        'VaR_reduction_%': round(float(var_red), 2),
        'ES_base':         round(float(es_base), 2),
        'ES_port':         round(float(es_port), 2),
        'ES_reduction_%':  round(float(es_red), 2)
    }

    return hedge_ticker, metrics

def diversification_abs(
    base_ticker: str,
    hedge_ticker: str,
    qty_base: int,
    qty_hedge: int,
    start_date: str,  # 'YYYYMMDD'
    end_date: str     # 'YYYYMMDD'
) -> tuple[str, dict]:
    """
    실제 주식 수량(qty) 기준으로 포트폴리오 VaR/ES 및 단일 대비 감소율을 계산.
    반환값: (hedge_ticker, {VaR_base, VaR_port, VaR_reduction_%})
    """
    # A) 가격 & 수익률
    price_df   = get_aligned_price_df(base_ticker, [hedge_ticker], start_date, end_date)
    returns_df = compute_daily_returns(price_df)

    # B) 단일 VaR (base only) — % 손실이 아니라 '포트폴리오 가치' 기준 손실
    base_vals = price_df[base_ticker] * qty_base
    base_rets = base_vals.pct_change().dropna()
    var_base  = -np.percentile(base_rets, 5)

    # C) 포트폴리오 VaR
    hedge_vals = price_df[hedge_ticker] * qty_hedge
    port_vals  = base_vals + hedge_vals
    port_rets  = port_vals.pct_change().dropna()
    var_port   = -np.percentile(port_rets, 5)

    # D) 절대 VaR 감소율
    var_red    = (var_base - var_port) / var_base * 100

    return hedge_ticker, {
        "VaR_base":        round(float(var_base), 4),
        "VaR_port":        round(float(var_port), 4),
        "VaR_reduction_%": round(float(var_red), 2)
    }


def mc_practice(base_ticker: str) -> list:
    """
    1) run_hedge_pipeline으로 최적 헷지 티커 뽑고
    2) diversification_effect 실행하여 리스크 감소(metrics) 계산
    3) [base_name, hedge_name, metrics] 반환
    """
    # 1년(365일) 전부터 오늘까지 기간
    end = date.today()
    start = end - timedelta(days=365)
    start_str, end_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')

    # ① 음(–)상관 헷지 후보 DataFrame
    df_neg = run_hedge_pipeline(base_ticker, n_candidates=100, lookback_days=365)
    if df_neg.empty:
        raise ValueError("음상관 헷지 후보가 없습니다.")

    # ② 최적 헷지 티커(첫 번째)
    hedge_ticker = str(df_neg.loc[0, "ticker"])

    # ③ 다각화 효과 계산
    _, metrics = diversification_effect(
        base_ticker=base_ticker,
        hedge_ticker=hedge_ticker,
        weight_base=0.5,
        weight_hedge=0.5,
        start_date=start_str,
        end_date=end_str
    )

    # ④ 종목명 매핑 후 결과 반환
    base_name  = find_name_by_ticker(base_ticker)
    hedge_name = find_name_by_ticker(hedge_ticker)
    return [base_name, hedge_name, metrics]

if __name__ == "__main__":
    # 예시 실행
    result = mc_practice("247540")
    print(result)

    # 예시 출력
    # {
    #   'VaR_base': 0.XX, 
    #   'VaR_port': 0.YY, 
    #   'VaR_reduction_%': ZZ.ZZ,
    #   'ES_base': 0.AA, 
    #   'ES_port': 0.BB, 
    #   'ES_reduction_%': CC.CC
    # }


# -------아래는 mc 하는거 처음 연습-----
# def simple_mc_hedge(
#     base_ticker: str,
#     hedge_ticker: str,
#     start_date: str,
#     end_date: str,
#     hedge_ratio: float = 1.0,
#     n_paths: int = 1000,
#     n_steps: int = 252
# ) -> dict:
#     """
#     Monte Carlo 시뮬레이션으로 헷지 P&L 분포를 구하고 95% VaR/ES 리턴
#     """
#     # 1) 가격을 가져와서 수익률 계산
#     price_df   = get_aligned_price_df(base_ticker, [hedge_ticker], start_date, end_date)
#     returns_df = compute_daily_returns(price_df.rename(columns={hedge_ticker: 'hedge', base_ticker: 'base'}))
    
#     # 2) 평균·공분산 계산
#     mu  = returns_df[['base','hedge']].mean().values      # shape (2,)
#     cov = returns_df[['base','hedge']].cov().values       # shape (2,2)
    
#     # 3) 시뮬레이션: (n_paths, n_steps, 2) 샘플
#     sims = np.random.multivariate_normal(mu, cov, size=(n_paths, n_steps))
    
#     # 4) 경로별 누적 P&L (ΔBase – hedge_ratio·ΔHedge)
#     pnl_paths = np.cumsum(sims[:,:,0] - hedge_ratio * sims[:,:,1], axis=1)
#     final_pnls = pnl_paths[:,-1]
    
#     # 5) VaR/ES 계산 (95% 신뢰수준)
#     var95 = -np.percentile(final_pnls, 5)
#     es95  = -final_pnls[final_pnls <= np.percentile(final_pnls, 5)].mean()
    
#     return {
#         "hedge_ratio": hedge_ratio,
#         "VaR_95%": var95,
#         "ES_95%": es95,
#         "simulated_pnls": final_pnls  # 필요시 히스토그램 등 추가 분석용
#     }

# # ── 사용 예시 ──
# if __name__=="__main__":
#     # 예: 2년치(≈504거래일) 데이터를 쓰고, 헷지비율 1.2로 1,000경로 시뮬
#     end    = date.today()
#     start  = end - timedelta(days=365*2)
#     result = simple_mc_hedge(
#         base_ticker="005930",
#         hedge_ticker="247540",
#         start_date=start.strftime("%Y%m%d"),
#         end_date=end.strftime("%Y%m%d"),
#         hedge_ratio=1.2,
#         n_paths=10000,
#         n_steps=252
#     )
#     print(f"헷지비율={result['hedge_ratio']}, 95% VaR={result['VaR_95%']:.4f}, 95% ES={result['ES_95%']:.4f}")
