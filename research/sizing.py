"""
research/sizing.py — Phase 5: turn positions (+1/-1/0) into target weights.

Steps:
  1. keep active tickers (todays_pos != 0)
  2. guard: too few names to hedge -> all zeros
  3. hedge: regress pos on [1, betas] across tickers, keep the residual
     -> sum(w) = 0 (dollar-neutral), sum(beta_ik * w) = 0 for every PC i
  4. sign check: count tickers whose sign flipped (hedge overriding the signal)
  5. scale to 1x gross: w / sum(|w|)
  6. put back into the full ticker list (flat tickers = 0)
"""

import numpy as np
import pandas as pd
import cvxpy as cp

def hedge_qp(y: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    y : (n,)    positions of active tickers, +1 / -1
    B : (n, K)  betas of active tickers (rows = tickers, cols = PCs)

    Returns w (n,): closest to y such that
        sum(w) = 0, B^T w = 0 (orthogonal), and sign(w_i) agrees with y_i (or w_i = 0).
    
    NOTES: use cp.sum... instead of Numpy functions because Numpy functions dont reliably understand cvxpy variables

    """
    n = len(y)

    # 1. variable: one weight per active ticker (variable means: the unknown we are solving for)
    w = cp.Variable(n)

    # 2. objective: stay as close as possible to the signal
    objective = cp.Minimize(cp.sum_squares(w-y))  # can proved by Lagrange multiplier that this is equivalent to minimizing sum((w-y)^2)

    # 3. constraints (fill these in)
    constraints = [
        # dollar-neutral:  sum of w equals 0
        cp.sum(w) == 0,
        # factor-neutral:  B^T @ w equals 0   (one equation per PC)
        B.T @ w == 0,
        # keep signs:      y_i * w_i >= 0     
        cp.multiply(y, w) >= 0
    ]

    # 4. solve
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # 5. check it actually solved
    if prob.status not in ("optimal", "optimal_inaccurate"): 
        return np.zeros(n)  # return all zeros if the optimization failed

    return w.value


def size_positions(pos: pd.Series, betas: pd.DataFrame) -> pd.Series:
    """
    pos   : (tickers,) +1 long, -1 short, 0 flat — from update_positions
    betas : (tickers x K) each ticker's beta to each PC — from compute_residuals (betas from compute_residuals are already transpose)

    Returns (tickers,) target weights: dollar-neutral, factor-neutral, sum(|w|) = 1.
    """
    w_full = pd.Series(0.0, index=pos.index)

    # 1. active tickers only
    active = pos[pos != 0].index                                # tickers where pos != 0
    y = pos.loc[active]                                    # pos for active tickers (n,)
    B = betas.loc[active]                                       # betas for active tickers (n x K)

    # 2. guard: need more names than columns in Z (1 + K)
    #    if not enough -> return w_full (all zeros)
    k = betas.shape[1]
    if len(active) <= k +1:
        return w_full
    
    # 3. hedge - using cvxpy to solve the regression problem with constraints

    w = hedge_qp(y.to_numpy(dtype=float), B.to_numpy(dtype=float))

    # 5. scale to 1x gross
    if np.sum(np.abs(w)) < 1e-9:
        return w_full
    else:
        pos_totrade = pd.Series(w/np.sum(np.abs(w)), index = active)
        w_full.loc[active] = pos_totrade
        return w_full


if __name__ == "__main__":
    # Toy test — work out by hand first.
    tickers = ['A', 'B', 'C', 'D', 'E']
    pos   = pd.Series([1, -1, 1, -1, 0], index=tickers)
    betas = pd.DataFrame({'PC1': [1.2, 0.8, 0.9, 1.1, 1.0]}, index=tickers)

    w = size_positions(pos, betas)
    print(w)
    print("sum w        :", w.sum())                          # ~0
    print("sum w*beta   :", (w * betas['PC1']).sum())         # ~0
    print("sum |w|      :", w.abs().sum())                    # 1
    # Expected ≈ A +0.167, B -0.167, C +0.333, D -0.333, E 0