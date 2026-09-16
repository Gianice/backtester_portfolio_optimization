# equity: 100 → 110 → 99 → 121
# peak is 110, trough is 99  →  max drawdown = 99/110 - 1 = -10%
TEST_CURVE = [
    {'real_date': '2020-01-01', 'todays_value': 100},
    {'real_date': '2020-01-02', 'todays_value': 110},
    {'real_date': '2020-01-03', 'todays_value':  99},
    {'real_date': '2020-01-04', 'todays_value': 121},
]

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# -- structure -- #
# equity_curve  =  list of dicts, one per bar
#                  [{'real_date': Timestamp, 'todays_value': float}, ...]

# all_equity_curves = {ticker: equity_curve}

def to_returns(equity_curve) -> pd.Series:
    s = pd.DataFrame(equity_curve).set_index('real_date')['todays_value']
    
    if not s.index.is_monotonic_increasing:
        raise ValueError(
            "equity curve is not chronologically ordered — "
            "the backtest processed bars out of order"
        )
    if s.index.has_duplicates:
        raise ValueError("duplicate dates in equity curve — a bar was processed twice")


    return s.pct_change().dropna()


def total_return(equity_curve) -> float:
    v = pd.DataFrame(equity_curve)['todays_value']
    return v.iloc[-1] / v.iloc[0] - 1.0


def sharpe(returns, rf=0.0) -> float:
    """Annualised Sharpe. rf is an annual rate; convert to daily first.
    Return np.nan if std is 0 — don't divide by zero and don't return inf."""
    
    if len(returns) < 2:
        return np.nan
    
    daily_rf = rf / (TRADING_DAYS)
    daily_r = returns.mean()
    std = returns.std()
    if not np.isfinite(std) or std < 1e-12:
        return np.nan
    
    annualize_sharpe = (daily_r - daily_rf) / std * np.sqrt(TRADING_DAYS)
    return annualize_sharpe


def max_drawdown(equity_curve):
    """Return (depth, peak_date, trough_date)."""
    
    df = pd.DataFrame(equity_curve).set_index('real_date')
    v = df['todays_value']
    
    dd = v/v.cummax() - 1
    
    trough_date = dd.idxmin()
    
    peak_date = v.loc[:trough_date].idxmax()
    return dd.min(), peak_date, trough_date
    
    
    
    
   
    
    
    
    
    
    
    