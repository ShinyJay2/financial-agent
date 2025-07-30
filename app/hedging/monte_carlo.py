
import numpy as np
from app.hedging.compute_returns import get_aligned_price_df, compute_daily_returns
from app.hedging.regression import run_hedge_pipeline

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


def monte_carlo(base_ticker):
    from datetime import date, timedelta
    end   = date.today()
    start = end - timedelta(days=365)

    df_neg = run_hedge_pipeline(base_ticker="247540", n_candidates=100, lookback_days=365)
    hedge_ticker = str(df_neg.loc[0, "ticker"]) #삼양식품이 나옴

    # 예: regression 결과 첫번째 티커(003230)를 바로 넣어서 실행
    hedge_ticker_result, result_df = diversification_effect(
        base_ticker=base_ticker,
        hedge_ticker=hedge_ticker,
        weight_base=0.5,
        weight_hedge=0.5,
        start_date=start.strftime('%Y%m%d'),
        end_date=end.strftime('%Y%m%d')
    )
    return hedge_ticker_result, result_df

# ── 실행 예시 ──
if __name__=='__main__':
# regression 함수임. 에코프로비엠 넣고 돌린 결과.
    hedge_ticker_result, result_df = monte_carlo("247540")
    print(f"선택된 헷지 티커: {hedge_ticker_result}")
    print(result_df)


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
