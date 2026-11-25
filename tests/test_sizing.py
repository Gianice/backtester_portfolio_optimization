import numpy as np
from research.sizing import size_portfolio

B_TOY = np.array([[1.2], [0.8], [1.0], [0.6]])


def test_toy_factor_neutral_and_values():
    signals = np.array([1, 1, -1, -1])
    w, status = size_portfolio(signals, B_TOY)
    assert status == "optimal"
    assert np.allclose(B_TOY.T @ w, 0, atol=1e-6)
    assert np.isclose(np.abs(w).sum(), 1.0)
    assert np.allclose(w, [0.2176, 0.2294, -0.2824, -0.2706], atol=1e-3)


def test_no_active_returns_zeros():
    w, status = size_portfolio(np.zeros(4), B_TOY)
    assert np.all(w == 0)
    assert status == "no active"


def test_signs_preserved_and_flat_is_zero():
    signals = np.array([1, 1, -1, -1, 0])
    B = np.vstack([B_TOY, [[0.9]]])
    w, status = size_portfolio(signals, B)
    assert status == "optimal"
    assert np.all(w[signals > 0] >= -1e-9)
    assert np.all(w[signals < 0] <= 1e-9)
    assert np.isclose(w[4], 0, atol=1e-9)


def test_one_sided_book_returns_zeros_not_nan():
    signals = np.array([1, 1, 0])
    B = np.array([[1.0], [0.8], [0.5]])
    w, status = size_portfolio(signals, B)
    assert not np.any(np.isnan(w))
    assert np.allclose(w, 0)