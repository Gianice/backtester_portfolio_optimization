"""
research/residuals.py — strip the factor component out of each stock.

For one window: regress every stock's returns on the K factor returns,
keep the residual, and cumulate it into X — the series the OU step fits.
"""

import numpy as np
import pandas as pd


def compute_residuals(win: pd.DataFrame, F: pd.DataFrame):
    """
    win : (days x tickers) raw returns for ONE window — same days and
          tickers that extract_factors used.
    F   : (days x K) factor returns from extract_factors.

    Returns
    -------
    alpha : Series (tickers,)           intercept per stock
    betas : DataFrame (tickers x K)     each stock's exposure to each factor
    residual  : DataFrame (days x tickers)  residual returns
    cumres     : DataFrame (days x tickers)  cumulative residual — what OU fits
    """
    X = np.column_stack([np.ones(len(win)), F.to_numpy()])
    coef, ssr, rank, sv = np.linalg.lstsq(X, win, rcond=None)
    residual = win - X @ coef
    
    # r = α + β·F
    #get r, α ,β
    
    residual = pd.DataFrame(residual, index= win.index, columns = win.columns)
    alpha = pd.Series(coef[0, :], index= win.columns, name='alpha')
    betas = pd.DataFrame(coef[1:].T, index = win.columns, columns=F.columns)
    cumres = residual.cumsum()

    return alpha, betas, residual, cumres


if __name__ == "__main__":
    from data.wide import load_wide_cached
    from research.factors import extract_factors, PCA_WINDOW

    prices, returns = load_wide_cached()

    # ONE window, just for testing
    win = returns.iloc[:PCA_WINDOW]

    F, loadings, w, k_info = extract_factors(win, None)
    win = win.loc[:, loadings.index]          # keep only tickers PCA kept

    alpha, betas, residual, cumres = compute_residuals(win, F)

    print(f"K = {F.shape[1]}, stocks = {win.shape[1]}")
    print("orthogonality  max|F.T @ eps| (~0): ", (F.T @ residual).abs().max().max())
    print("last-day X     max|cumres[-1]| (~0):", cumres.iloc[-1].abs().max())
    print("mean PC1 beta  (> 0):               ", betas['PC1'].mean())
    print("residual std / raw std (median):    ", (residual.std() / win.std()).median())