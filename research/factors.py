
"""
research/factors.py — PCA factor extraction.

Produces the market proxy to subtract. Contributes NO return forecast:
this is the risk model, it says what to hedge away. All alpha is in ou.py.
"""

import numpy as np
import pandas as pd

PCA_WINDOW = 252          # must be >> N or the matrix is rank-deficient


def marchenko_pastur_bound(n_stocks: int, n_days: int) -> float:
    """
    The largest eigenvalue that pure noise can produce, 
    given how many stocks you have and how many days of data 
    — anything below it could be luck, anything above it is a real factor.
    """
    q = n_stocks / n_days
    return (1 + np.sqrt(q)) ** 2


def choose_k(eigenvalues: np.ndarray, n_stocks: int, n_days: int) -> dict:
    """
    Three criteria. Compute all three, report the disagreement.

    Returns {'variance_55': int, 'marchenko_pastur': int}

      variance_55       how many PCs to reach 55% cumulative variance.
                        eigenvalues sum to n_stocks, so share = lam / n_stocks.
      marchenko_pastur  count above the noise bound. The one to trust.
    """
    res = {}
    ## Calculate variance_55
    shares = eigenvalues / n_stocks
    cum = shares.cumsum()
    res['variance_55'] = len(cum[cum < 0.55]) + 1
    
    mp = int((eigenvalues > marchenko_pastur_bound(n_stocks, n_days)).sum())
    
    res['mp'] = mp
    return res
    


def extract_factors(win: pd.DataFrame, k: int):
    ##win is 252 * 87 dataframe: 252 days as rows, all tickers as column
    """
    returns : (days x tickers) for ONE estimation window. Trailing only —
              never include the day you are trading.

    Returns
    -------
    F        : (days x k)      factor return series — what the regression uses
    loadings : (tickers x k)   eigenvector weights — needed later for
                               factor neutrality in build_weights
    eigvals  : (n_stocks,)     descending — for choose_k and the sanity checks

    Four gotchas, in order of damage done:
      1. CORRELATION, not covariance. Standardise first. Otherwise one
         5x-volatility name hijacks PC1 (loading 0.96 vs 0.06 for the rest)
         and you hedge that stock instead of the market.
      2. eigh returns ASCENDING. Reverse both w and v, together.
      3. Eigenvectors are COLUMNS. v[:, 0] is the first one;
         v[0, :] is stock 0's loadings across all components.
      4. SIGN IS ARBITRARY and numpy can flip it between consecutive
         windows. Force a convention every window, or the factor series
         silently reverses mid-backtest and every beta after it is wrong.
    """
    
    win = win.loc[:, win.std > 0]  
    
    R = win.to_numpy()
    
    n_days, n_stocks = R.shape

    Rs = (R - R.mean(axis = 0)) / R.std(axis = 0) ##standardize to have variance = 1
    
    C = np.corrcoef(Rs.T)           ##correlation matrix
    w, v = np.linalg.eigh(C) 
    ## -> return eigenvalue(variance) + 
    # eigenvector(corresponding portfolio weight) (both in ascending order)
    
    w, v = w[::-1], v[:, ::-1] ## make it descending
    
    k_info = choose_k(w, n_stocks, n_days)
    if k is None:
        k = max(1, k_info['mp'])
    # Sign convention fix: eigh() returns v or -v arbitrarily (both solve Cv = λv equally well).
    # Without this, the same factor could flip sign across rolling windows for no real reason,
    # Force a consistent sign (here: majority loading positive) 
    for j in range(k):
        if v[:, j].sum() < 0: v[:, j] = -v[:, j]
        
    cols = [f'PC{j + 1}' for j in range(k)]
    F = pd.DataFrame(Rs @ v[:, :k], index = win.index, columns=cols)
    loadings = pd.DataFrame(v[:, :k], index = win.columns, columns = cols)
    return F, loadings, w, k_info

def rolling_factors(returns: pd.DataFrame, window: int = PCA_WINDOW, k: int | None = None) -> dict:
    """
    Refit PCA on every trailing window.

    Keyed by trade date t. The window is returns[t-window : t], which
    EXCLUDES day t itself, so the factors for t only use data before t.
    """
    out = {}
    for t in range(window, len(returns)):
        win = returns.iloc[t - window:t]
        F, loadings, eigvals, k_info = extract_factors(win, k)
        ## each date has one info
        out[returns.index[t]] = {
            'F': F,
            'loadings': loadings,
            'eigvals': eigvals,
            'k_info': k_info,
        }
    return out



if __name__ == "__main__":
    import os
    from data.wide import load_wide_cached

    prices, returns = load_wide_cached()
    results = rolling_factors(returns)

    k_table = pd.DataFrame({d: r['k_info'] for d, r in results.items()}).T
    print(f"windows: {len(results)}")
    print(k_table.describe().loc[['min', '50%', 'max']])

    # Save ONE real window as the pytest fixture (last trade date)
    last = results[max(results)]
    os.makedirs("tests/fixtures", exist_ok=True)
    np.savez(
        "tests/fixtures/pca_window_sample.npz",
        eigvals=last['eigvals'],                         # (N,) descending
        loadings=last['loadings'].to_numpy(),            # (N, k)
        tickers=np.array(last['loadings'].index),        # N tickers, matching row order
    )
    print("fixture saved")


    
    
    