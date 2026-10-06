# tests/test_portfolio.py
import pandas as pd
import pytest
from engine.backtest_statistical_arbitrage import run_portfolio_backtest


#------EQUITY CURVES AND WEIGHT LOG STRUCTURE ----------#
# equity_curves.append({'real_date': date, 
#                               'todays_value': cash + (target_shares*px).sum(),
#                               'turnover': turnover,
#                               'cost' : cost})
#         weight_log[date] = w


CAP = 100000

def _toy():
    '''
    The prices are arbitrary
    '''
    
    idx = pd.date_range('2024-01-12', periods=4)
    prices = pd.DataFrame({'AAPL': [100.,105.,140.,120.], 'MSFT': [100., 95., 90., 105. ]}, index=idx)
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

def  test_constant_weights_zero_turnover() :
    '''
    Holding the same weight costs nothing after the opening trade
    Expected turnover: 0 (shift leaves day 1 flat), 1.0(book opens) then 0
    Catches: cost charge on |w| instead of |w - prev_w|, or prev_w never updated
    '''
    
    idx, prices = _toy()
    weights = pd.DataFrame({'AAPL':[.5]*4, 'MSFT':[-.5]*4}, index = idx, columns= prices.columns)
    ec, wl = run_portfolio_backtest(prices, weights)
    
    turnover = (r['turnover'] for r in ec)
    
    assert turnover[0] == 0.0
    assert turnover[1] == pytest.approx(1.0)
    assert all(r == 0.0 for r in turnover[2:])
    

def test_dollar_neutral_immune_to_market():
    """
    Equal long and short legs on two stocks moving at the same % -> flat equity
    Exp: Both rises at 10% then 10%, +5000 on the long cancel -5000 on the short -> breakeven
    """
    
    idx = pd.date_range('2024-01-02', period = 3)
    prices = pd.DataFrame({"APPL": [100, 110, 121], 'B': [50, 55, 60.5]}, index = idx)
    weight = pd.DataFrame({'APPL': [.5, .5, .5], 'B': [-.5, -.5, -.5]}, index = idx)
    
    equity_curve, _ = run_portfolio_backtest(prices, weight)
    
    for row in equity_curve:
        assert row['todays_value'] == pytest.approx(CAP)
        

def test_costs_scale_linearly():
    '''
    Doubling cost_bps doubles total cost;
    '''
    idx, prices = _toy()
    
    weights = pd.DataFrame({'AAPL': [.5] * 4, 'MSFT': [-.5] * 4}, index= idx)
    
    ec5, wl5 = run_portfolio_backtest(prices, weights, cost_bps= 5.0)
    ec10, wl10 = run_portfolio_backtest(prices, weights, cost_bps= 10.0)
    
    c5 = sum(row['cost'] for row in ec5)
    c10 = sum(row['cost'] for row in ec10)
    
    assert c10 == pytest.approx(2*c5)
    

def execution_lag_shifts():
    """
    A weight decided on day t is executed in day t+1
    """
    
    idx, prices = _toy()
    weights = pd.DataFrame({"APPL": [0., 0., 1., 0.], 
                            "MSFT": [0., 0., 0., 0,]}, index = idx)
    
    _, weight_log = run_portfolio_backtest(prices, weights)
    held = pd.DataFrame(weight_log).T
    
    assert held['APPL'].iloc[2] == 0 ##signal day, not yet traded
    assert held['APPL'].iloc[3] == 1 ## traded next day
    
    
    
    

    
    
    
    
    
    
    
        
    
        
