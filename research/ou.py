"""
research/ou.py — fit an OU process to each stock's cumulative residual,
filter for genuine mean reversion, and compute the s-score.

This is where the alpha lives: factors.py says what to hedge,
residuals.py isolates the stock-specific drift, ou.py asks whether
that drift comes back — and how stretched it is right now.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ADF_PVALUE = 0.05
HALF_LIFE_MIN = 1      # days
HALF_LIFE_MAX = 30     # days


def fit_ou(cumres: pd.DataFrame) -> pd.DataFrame:
    """
    cumres : (days x tickers) cumulative residual X from compute_residuals.

    Returns a DataFrame indexed by ticker with columns:
      a, b, kappa, half_life, m, sigma_eq, s, adf_p, tradeable
    """
    rows = {}
    for ticker in cumres.columns:
        x = cumres[ticker].to_numpy()
        
        x_today = x[:-1]
        x_next = x[1:]
        
        # coef, ssr, rank, sv = np.linalg.lstsq(A, y, rcond=None)
        # coef: the coefficients, one per column of A. coef = [a, b]. 
        # ssr: sum of squared residuals, a single number. Not the residual series.
        # rank: rank of A. Should be 2 here.
        # sv: singular values of A, for diagnostics only.
        X = np.column_stack([np.ones(len(x_today)), x_today])
        coef, *_ = np.linalg.lstsq(X, x_next, rcond= None)
        a, b = coef
        residual = x_next - (a + b * x_today)

        # 4. OU parameters (per-day units):
        #if b <= 0 or b >= 1, ln(b) and these formulas break.
        # no signal to trade when b <= 0 or b >= 1
        if (b <=0) or (b >=1) :
            kappa = np.nan
            half_life = np.nan
            m = np.nan
            sigma_eq = np.nan
        else:
            kappa     = -np.log(b)
            half_life = np.log(2) / kappa
            m         = a / (1 - b)
            sigma_eq  = np.sqrt(np.var(residual) / (1 - b**2) )
        

        # 5. s = (today's cumres − equilibrium) / normal distance from equilibrium
        #      = (x[-1] − m) / σ_eq
        s = (x[-1] - m) / sigma_eq
        
        # 6. ADF test on x. ONLY NEED p-value.
        p = adfuller(x, maxlag=1, regression='c')[1] 
        # adfuller returns a tuple; p-value is index [1].
    

        rows[ticker] = {
            'a': a,
            'b': b,
            'kappa' : kappa,
            'half_life' : half_life,
            'm' : m,
            'sigma_eq' : sigma_eq,
            'adf_p': p,
            's': s
        }

    out = pd.DataFrame(rows).T

    # 7. tradeable = ADF passes AND kappa > 0 AND half-life inside the bounds
    #    NaN comparisons evaluate to False, so b outside (0, 1) is excluded automatically.
    out['tradeable'] = ((out['adf_p'] < ADF_PVALUE) 
                       & (out['kappa'] > 0)
                       & (out['half_life'] >= HALF_LIFE_MIN)
                       & (out['half_life'] <= HALF_LIFE_MAX)
                        )

                       
    return out

if __name__ == "__main__":
    from data.wide import load_wide_cached
    from research.factors import extract_factors, PCA_WINDOW
    from research.residuals import compute_residuals

    prices, returns = load_wide_cached()

    win = returns.iloc[-PCA_WINDOW:]
    F, loadings, w, k_info = extract_factors(win, None)
    win = win.loc[:, loadings.index]
    alpha, betas, residual, cumres = compute_residuals(win, F)

    ou = fit_ou(cumres)
    t = ou[ou['tradeable']]

    print(f"tradeable: {len(t)} / {len(ou)}")
    print(f"failed b in (0,1):   {ou['kappa'].isna().sum()}")
    print(f"failed ADF:          {(ou['adf_p'] >= ADF_PVALUE).sum()}")
    print(f"median half-life (tradeable): {t['half_life'].median():.1f} days")
    print(f"would open (|s| > 1.25):      {(t['s'].abs() > 1.25).sum()}")
    print("\n", t[['s', 'half_life', 'adf_p']].sort_values('s'))