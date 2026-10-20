import numpy as np
import pandas as pd


'''
Checks for residual correlation in the factor model regression. The residual of the regression of return on PCA factors should be uncorrelated. 
If they are correlated, it indicates that the PCA factors are not sufficient to explain the returns, and there is some structure left in the residuals.
'''


def residual_correlation_check(res: pd.DataFrame, top=10):
    """res: (days x tickers) daily residuals from the factor regression."""
    C = res.corr()                                   # 87 x 87
    n_days, n_stocks = res.shape

    # 1. off-diagonal values only (upper triangle, no diagonal)
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    #   F T T
    #   F F T
    #   F F F

    ##1. off [one long column] : correlation matrix values for all pairs of tickers, excluding the diagonal and lower triangle.
    off = C.where(mask).stack()                      # Series indexed by (ticker_i, ticker_j)

    # 2. summary vs noise level
    ## noise = 1/√T is the size of correlation two unrelated series show by chance with T days of data
    noise = 1 / np.sqrt(n_days)
    # print mean, std, noise, count of |off| > 0.2
    n_big= np.sum(off.abs() > 0.2)
    mean = off.mean()
    median = off.median()
    std = off.std()
    count = len(off)


    # 3. top pairs
    top_pairs = off.abs().sort_values(ascending=False).head(top)

    # 4. eigenvalues vs Marchenko–Pastur
    mp = (1 + np.sqrt(n_stocks / n_days)) ** 2
    w = np.linalg.eigvalsh(C)[::-1]
    n_above = np.sum(w > mp)

    print(f"MP bound:            {mp:.2f}")
    print(f"largest eigenvalue:  {w[0]:.2f}")
    print(f"top 5 eigenvalues:   {np.round(w[:5], 2)}")
    print(f"above MP bound:      {n_above}")
    print(f"mean correlation:    {mean:.4f}")
    print(f"median correlation:  {median:.4f}")
    print(f"std correlation:     {std:.4f}")
    print(f"count of significant correlations: {n_big}")
    print(f"top {top} correlations:\n{top_pairs}")
    return C

if  __name__ == "__main__":
    from data.wide import load_wide_cached
    from research.factors import extract_factors, PCA_WINDOW
    from research.residuals import compute_residuals

    print("diagnostics: start")

    prices, returns = load_wide_cached()

    # ONE window, just for testing
    win = returns.iloc[:PCA_WINDOW]

    F, loadings, w, k_info = extract_factors(win, None)
    win = win.loc[:, loadings.index]          # keep only tickers PCA kept

    alpha, betas, residual, cumres = compute_residuals(win, F)
    residual_correlation_check(residual)