import numpy as np
import pytest

# 'loadings' structure
#           PC0      PC1      PC2     ...    PC86
# AAPL   [ 0.14,    0.31,   -0.02,   ...,    0.00 ]
# MSFT   [ 0.13,    0.28,    0.01,   ...,    0.00 ]
# NVDA   [ 0.15,   -0.40,    0.22,   ...,    0.00 ]
# XOM    [ 0.09,   -0.15,   -0.35,   ...,    0.01 ]
# CVX    [ 0.08,   -0.12,   -0.30,   ...,    0.01 ]
#  ...     ...       ...      ...             ...

@pytest.fixture
def pca_window_fixture():
    data = np.load("tests/fixtures/pca_window_sample.npz", allow_pickle=True)
    
    '''
    CREATE THE FILE USING BELOW COMMAND to get the above code run smoothly:
    np.savez(
    "tests/fixtures/pca_window_sample.npz",
    eigvals=w,          # the (87,) eigenvalues from one real window
    loadings=v,         # the (87, 87) eigenvectors from that same window
    tickers=np.array(ticker_list),   # the 87 ticker strings, in matching order
    )
    '''
    
    return data["eigvals"], data["loadings"], data["tickers"]


def test_eigenvalues_sum_to_n(pca_window_fixture):
    """Standardisation is correct: trace(C) = N for a correlation matrix,
    so eigenvalues must sum to N (one unit of variance per stock)."""
    eigvals, _, tickers = pca_window_fixture
    assert eigvals.sum() == pytest.approx(len(tickers), rel=1e-6)


def test_eigenvalues_descending(pca_window_fixture):
    """The reversal step worked — eigvals[0] is the largest."""
    eigvals, _, _ = pca_window_fixture
    assert np.all(np.diff(eigvals) <= 0), "eigenvalues are not sorted descending"
    ##note: np.diff compute 'next - current', not 'current - next'



def test_pc1_is_one_signed(pca_window_fixture):
    """PC1 is the market: most stocks load one way, and any that load the
    other way must be weakly attached — smaller than a typical loading.
    A large wrong-signed loading would mean PC1 is a spread, not the market.
    On the saved window: 13 defensives load negatively, largest |-0.064|,
    vs majority median 0.099."""
    _, loadings, tickers = pca_window_fixture
    pc1 = loadings[:, 0]

    majority = (pc1 > 0) if (pc1 > 0).mean() > 0.5 else (pc1 < 0)
    assert majority.mean() > 0.75, "PC1 has no dominant sign — not a market factor"

    typical = np.median(np.abs(pc1[majority]))
    worst_wrong = np.abs(pc1[~majority]).max() if (~majority).any() else 0.0
    assert worst_wrong < typical, (
        f"wrong-signed loading {worst_wrong:.3f} >= typical {typical:.3f} — "
        "PC1 looks like a spread factor"
    )


def test_pc1_top_bottom_loadings_eyeball(pca_window_fixture, capsys):
    """Not a pass/fail assertion — prints PC1's extremes so a human can
    eyeball that the largest loadings are big liquid names, not junk.
    Run with `pytest -s` to see the output."""
    _, loadings, tickers = pca_window_fixture
    pc1 = loadings[:, 0]
    order = np.argsort(np.abs(pc1))[::-1]
    print("\nPC1 top 5 by |loading|:")
    for i in order[:5]:
        print(f"  {tickers[i]:<6} {pc1[i]:+.4f}")
    print("PC1 bottom 5 by |loading|:")
    for i in order[-5:]:
        print(f"  {tickers[i]:<6} {pc1[i]:+.4f}")