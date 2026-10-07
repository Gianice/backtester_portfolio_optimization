# tests/test_backtest_statistical_arbitrage.py
import pandas as pd
import pytest
from engine.backtest_statistical_arbitrage import run_portfolio_backtest


#------EQUITY CURVES AND WEIGHT LOG STRUCTURE ----------#   # CHANGED
# equity_curves.append({'real_date': date,
#                       'todays_value': value,
#                       'turnover': turnover,
#                       'cost': cost,
#                       'net_exposure': ...,      # actual book, w.sum()
#                       'n_traded': ...})         # names traded that day
# weight_log[date] = shares * px / value          # ACTUAL weights, not targets


CAP = 100000

def _toy():
    '''
    The prices are arbitrary
    '''
    idx = pd.date_range('2024-01-12', periods=4)
    prices = pd.DataFrame({'AAPL': [100., 105., 140., 120.],
                           'MSFT': [100., 95., 90., 105.]}, index=idx)
    return idx, prices


def _flat(n=4):                                                # NEW
    '''
    Prices never move -> no drift, so any turnover must come from the target changing
    '''
    idx = pd.date_range('2024-01-12', periods=n)
    prices = pd.DataFrame({'AAPL': [100.] * n, 'MSFT': [100.] * n}, index=idx)
    return idx, prices


def test_zero_weights_zero_pnl():
    '''
    No positions -> equity never moves and no cost is ever charged
    cost_bps is deliberately nonzero, so this proves costs come from trading
    not from simply existing
    '''
    idx, prices = _toy()
    weights = pd.DataFrame(0., index=idx, columns=prices.columns)

    ec, wl = run_portfolio_backtest(prices, weights, cost_bps=5.)

    for row in ec:
        assert row['todays_value'] == 100_000.0
        assert row['cost'] == 0.0


def test_constant_weights_zero_turnover():                     # CHANGED: _flat, not _toy
    """
    Holding the same weight costs nothing after the opening trade.
    Flat prices -> no drift, so this isolates prev_w being updated.
    Expected turnover: 0 (lag leaves day 1 flat), 1.0 (book opens), then 0.
    Catches: cost charged on |w| instead of |w - prev_w|, or prev_w never updated.
    """
    idx, prices = _flat()
    weights = pd.DataFrame({'AAPL': [.5] * 4, 'MSFT': [-.5] * 4}, index=idx)

    ec, _ = run_portfolio_backtest(prices, weights)
    turnover = [r['turnover'] for r in ec]

    assert turnover[0] == 0.0
    assert turnover[1] == pytest.approx(1.0)
    assert all(t == 0.0 for t in turnover[2:])


def test_dollar_neutral_immune_to_market():
    """
    Equal long and short legs on two stocks moving the same % -> flat equity.
    Both rise 10% then 10%: the long's gain cancels the short's loss every day.
    """
    idx = pd.date_range('2024-01-02', periods=3)
    prices = pd.DataFrame({'AAPL': [100., 110., 121.],
                           'B':    [ 50.,  55.,  60.5]}, index=idx)
    weights = pd.DataFrame({'AAPL': [.5] * 3, 'B': [-.5] * 3}, index=idx)

    ec, _ = run_portfolio_backtest(prices, weights, cost_bps=0.)

    for row in ec:
        assert row['todays_value'] == pytest.approx(CAP)


def test_long_position_earns_price_return():
    """
    The one test where P&L must be NONZERO — without it, an engine that
    never moves equity would pass everything above.
    Fully long AAPL from day 2 (after lag): equity tracks AAPL's price
    from the fill. Fill at 105, mark at 140 -> 100k * 140/105.
    """
    idx, prices = _toy()
    weights = pd.DataFrame({'AAPL': [1.] * 4, 'MSFT': [0.] * 4}, index=idx)

    ec, _ = run_portfolio_backtest(prices, weights, cost_bps=0.)

    assert ec[2]['todays_value'] == pytest.approx(CAP * 140 / 105)


def test_costs_scale_linearly():
    '''
    Doubling cost_bps doubles total cost (to within compounding:
    costs shrink capital, which shrinks later trades slightly)
    '''
    idx, prices = _toy()
    weights = pd.DataFrame({'AAPL': [.5] * 4, 'MSFT': [-.5] * 4}, index=idx)

    ec5, _ = run_portfolio_backtest(prices, weights, cost_bps=5.0)
    ec10, _ = run_portfolio_backtest(prices, weights, cost_bps=10.0)

    c5 = sum(row['cost'] for row in ec5)
    c10 = sum(row['cost'] for row in ec10)

    assert c5 > 0
    assert c10 == pytest.approx(2 * c5, rel=1e-3)


def test_execution_lag_shifts():
    """
    A weight decided on day t is executed on day t+1
    """
    idx, prices = _toy()
    weights = pd.DataFrame({"AAPL": [0., 0., 1., 0.],
                            "MSFT": [0., 0., 0., 0.]}, index=idx)

    _, weight_log = run_portfolio_backtest(prices, weights)
    held = pd.DataFrame(weight_log).T

    assert held['AAPL'].iloc[2] == 0                       # signal day, not yet traded
    assert held['AAPL'].iloc[3] == pytest.approx(1)        # CHANGED: actual weight, float


def test_band_zero_reproduces_daily_rebalance():               # NEW
    """
    Regression: band=0 must give the pre-band engine's turnover exactly.
    The numbers are the old engine's output on _toy.
    """
    idx, prices = _toy()
    weights = pd.DataFrame({'AAPL': [.5] * 4, 'MSFT': [-.5] * 4}, index=idx)

    ec, _ = run_portfolio_backtest(prices, weights, band=0.0)
    turnover = [r['turnover'] for r in ec]

    assert turnover == pytest.approx([0.0, 1.0, 0.161764705882353, 0.19718309859154934])


def test_band_holds_small_drift_trades_large():                # NEW
    """
    band=0.2 on a 0.5 target -> allowed range 0.40–0.60.
    +5%  : AAPL weight 0.512 -> inside, no trade
    +60% : AAPL weight 0.615 -> outside, trade back to target
    """
    idx = pd.date_range('2024-01-12', periods=5)
    prices = pd.DataFrame({'AAPL': [100., 100., 100., 105., 160.],
                           'MSFT': [100.] * 5}, index=idx)
    weights = pd.DataFrame({'AAPL': [.5] * 5, 'MSFT': [-.5] * 5}, index=idx)

    ec, _ = run_portfolio_backtest(prices, weights, band=0.2)
    turnover = [r['turnover'] for r in ec]

    assert turnover[3] == 0.0
    assert turnover[4] > 0.0