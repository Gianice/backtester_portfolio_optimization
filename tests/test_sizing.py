"""
tests/test_sizing.py — checks for research/sizing.py

What sizing must guarantee:
  1. dollar-neutral      sum(w) = 0
  2. factor-neutral      sum(w * beta_k) = 0 for every PC k   (the hedge)
  3. 1x gross            sum(|w|) = 1
  4. signs kept          w_i has the same sign as the signal, or is 0
  5. flat tickers stay 0, and every ticker is in the output
"""
import numpy as np
import pandas as pd
import pytest

from research.sizing import size_positions, hedge_qp

ATOL = 1e-6   # cvxpy solutions are accurate to roughly 1e-6 – 1e-8


def _lstsq_residual(y, B):
    """The plain regression hedge: residual of y on [1, B]."""
    Z = np.column_stack([np.ones(len(y)), B])
    c, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ c


# ---------------------------------------------------------------- 1. toy case
def test_toy_case_matches_hand_calculation():
    """
    pos   = [+1, -1, +1, -1, 0]       E is flat
    beta  = [1.2, 0.8, 0.9, 1.1, 1.0]
    hedge -> [0.6, -0.6, 1.2, -1.2]   then / 3.6
          -> [1/6, -1/6, 1/3, -1/3, 0]
    """
    tickers = ['A', 'B', 'C', 'D', 'E']
    pos = pd.Series([1, -1, 1, -1, 0], index=tickers)
    betas = pd.DataFrame({'PC1': [1.2, 0.8, 0.9, 1.1, 1.0]}, index=tickers)

    w = size_positions(pos, betas)

    expected = pd.Series([1/6, -1/6, 1/3, -1/3, 0.0], index=tickers)
    assert list(w.index) == tickers                       # every ticker present
    assert np.allclose(w, expected, atol=ATOL)
    assert np.isclose(w.sum(), 0, atol=ATOL)              # dollar-neutral
    assert np.isclose((w * betas['PC1']).sum(), 0, atol=ATOL)   # factor-neutral
    assert np.isclose(w.abs().sum(), 1, atol=ATOL)        # 1x gross
    assert w['E'] == 0                                    # flat stays flat


# ------------------------------------------- 2. QP == regression when no flips
def test_qp_equals_regression_when_no_sign_flips():
    """Without a binding sign constraint, the QP answer is the regression residual."""
    y = np.array([1, -1, 1, -1], dtype=float)
    B = np.array([[1.2], [0.8], [0.9], [1.1]])

    w_qp = hedge_qp(y, B)
    w_reg = _lstsq_residual(y, B)

    assert np.all(y * w_reg >= -ATOL)                     # confirm: regression has no flips here
    assert np.allclose(w_qp, w_reg, atol=ATOL)            # -> QP must match it
    assert np.allclose(w_qp, [0.6, -0.6, 1.2, -1.2], atol=ATOL)


# --------------------------------------------------------- 3. never flips signs
def test_qp_never_flips_a_sign():
    """
    Here the plain regression would flip A from long to short (w_A = -0.2).
    The QP must keep every weight on its signal's side (or set it to 0),
    while still being dollar- and factor-neutral.
    """
    y = np.array([1, -1, 1, -1, 1], dtype=float)
    B = np.array([[1.5], [0.5], [1.0], [1.0], [1.0]])

    w_reg = _lstsq_residual(y, B)
    assert w_reg[0] < 0                                   # confirm: regression flips A

    w_qp = hedge_qp(y, B)
    assert np.all(y * w_qp >= -ATOL)                      # no flips
    assert np.isclose(w_qp.sum(), 0, atol=ATOL)           # dollar-neutral
    assert np.allclose(B.T @ w_qp, 0, atol=ATOL)          # factor-neutral


# ------------------------------------------------------------------ 4. guard
def test_guard_too_few_names_returns_zeros():
    """K = 1 factor -> need more than 1 + K = 2 active names. Two names -> sit out."""
    tickers = ['A', 'B', 'C']
    pos = pd.Series([1, -1, 0], index=tickers)
    betas = pd.DataFrame({'PC1': [1.2, 0.8, 1.0]}, index=tickers)

    w = size_positions(pos, betas)
    assert list(w.index) == tickers
    assert (w == 0).all()


def test_all_flat_returns_zeros():
    tickers = ['A', 'B', 'C', 'D']
    pos = pd.Series([0, 0, 0, 0], index=tickers)
    betas = pd.DataFrame({'PC1': [1.2, 0.8, 0.9, 1.1]}, index=tickers)

    assert (size_positions(pos, betas) == 0).all()


# ------------------------------------------------ 5. the hedge actually works
def test_factor_only_day_has_zero_pnl():
    """
    If every stock moves ONLY by its factor part (r = B @ f),
    a hedged book must make nothing: sum(w * r) = 0.
    Uses 2 factors and 12 random stocks.
    """
    rng = np.random.default_rng(0)
    tickers = [f'T{i}' for i in range(12)]
    betas = pd.DataFrame(rng.normal(1.0, 0.3, size=(12, 2)),
                         index=tickers, columns=['PC1', 'PC2'])
    pos = pd.Series(rng.choice([1, -1], size=12), index=tickers)

    w = size_positions(pos, betas)
    assert w.abs().sum() > 0                              # it actually traded

    f = np.array([0.02, -0.01])                           # factor moves today
    r = pd.Series(betas.to_numpy() @ f, index=tickers)    # factor-only returns
    assert np.isclose((w * r).sum(), 0, atol=ATOL)


def test_pnl_comes_only_from_residuals():
    """Add stock-specific moves: book P&L must equal sum(w * residual)."""
    rng = np.random.default_rng(1)
    tickers = [f'T{i}' for i in range(12)]
    betas = pd.DataFrame(rng.normal(1.0, 0.3, size=(12, 2)),
                         index=tickers, columns=['PC1', 'PC2'])
    pos = pd.Series(rng.choice([1, -1], size=12), index=tickers)

    w = size_positions(pos, betas)

    f = np.array([0.02, -0.01])
    eps = pd.Series(rng.normal(0, 0.01, size=12), index=tickers)
    r = pd.Series(betas.to_numpy() @ f, index=tickers) + eps

    assert np.isclose((w * r).sum(), (w * eps).sum(), atol=ATOL)