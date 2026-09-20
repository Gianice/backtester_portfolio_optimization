"""
engine/portfolio_backtest.py — Phase 0: the new engine.

Your existing engine/backtest.py holds ONE ticker and flips between
cash and fully-invested. It cannot express "long 30 names, short 30 names,
each at 1.6% of capital". This file replaces it.

Do this FIRST, before any PCA. If the engine is wrong every number that
comes after it is wrong, and you will spend a week blaming the alpha.
"""
import numpy as np
import pandas as pd
TRADING_DAYS = 252

def run_portfolio_backtest(prices: pd.DataFrame,
                           weights: pd.DataFrame,
                           initial_capital: float = 100_000.0,
                           cost_bps: float = 0.0,
                           execution_lag: int = 1):
    """
    prices  : (days x tickers) PX_LAST, wide.
    weights : (days x tickers) TARGET weights, same index and columns.
              Row t = what you want to hold at the close of day t.
              Positive = long, negative = short.
              Row sums to ~0 (dollar neutral); abs() sums to ~1 (1x gross).

    Returns (equity_curve, weight_log, stats).

    The three things that make this different from backtest.py
    ---------------------------------------------------------
    1. WEIGHTS ARE TARGETS, NOT SIGNALS.
       backtest.py: signal 1 -> go all in, signal -1 -> go all out.
       here:        weight -0.016 -> hold -0.016 * equity of AAPL, today.
       Every bar you rebalance from where you are to where you want to be.

    2. COSTS COME FROM TURNOVER, NOT FROM TRADE COUNT.
       cost_t = cost_bps/10000 * sum(|w_t - w_{t-1}|) * equity_t
       Holding the same book two days running costs zero. Flipping the whole
       book costs 2x gross. This is why the s-score exit bands are
       asymmetric — every unnecessary round trip is paid for here.

    3. SHARES CAN BE NEGATIVE.
       Shorting credits cash: sell 100 AAPL at 150 and cash goes UP 15,000
       while shares go to -100. Equity is still cash + shares * price,
       the same line as before, but now shares * price can be negative and
       cash can exceed equity. If you guard with `shares > 0` anywhere in
       here you have re-introduced the long-only assumption.

    Execution lag: weights are computed from data up to the close of t, so
    they cannot be traded until t+1. shift(execution_lag) BEFORE the loop,
    exactly as you did with signals. Forgetting this is the single most
    common reason a stat arb backtest shows a Sharpe of 4.
    
    --price structure:
                AAPL    MSFT    INTC     AMD
    2024-01-02  185.0   370.0    47.0   140.0
    2024-01-03  186.0   372.0    46.5   139.0
    2024-01-04  184.0   375.0    47.5   142.0
    2024-01-05  188.0   371.0    47.0   141.0
    
    --weight structure:
                AAPL    MSFT    INTC     AMD
    2024-01-02   0.00    0.00    0.00    0.00
    2024-01-03   0.25    0.25   -0.25   -0.25
    2024-01-04   0.25    0.25   -0.25   -0.25
    2024-01-05   0.00    0.00    0.00    0.00
    """
    equity_curves = []
    weight_log = {}
    weights = weights.reindex(prices.index).shift(execution_lag).fillna(0.0)
    cash = initial_capital; shares = pd.Series(0.0, index = prices.columns)
    
    prev_w = pd.Series(0.0, index = prices.columns)
    
    for date in prices.index:
        px = prices.loc[date]
        w = weights.loc[date]
        
        '''
        Definition: 
        equity: yesterday's position, valued at today's price
        turnover: change in position (today vs yesterday)
        
        shares: shares before today's trade; target_shares : shares traded today
        '''
        equity = cash + (px * shares).sum()
        turnover = (w - prev_w).abs().sum()
        cost = turnover * equity * cost_bps / 10000
        target_shares = w * equity / px
        cash = cash - ((target_shares - shares) * px).sum() - cost

        shares = target_shares
        prev_w = w
        
        equity_curves.append({'real_date': date, 'todays_value': cash + (target_shares*px).sum()})
        weight_log[date] = w
        
    return equity_curves



def compute_portfolio_stats(equity_curve, weight_log, rf = 0.0) -> dict:
    """
    Sharpe, max drawdown, CAGR — same as research/metrics.py — PLUS the
    three that only exist for a long/short book:

      avg_gross      mean of abs(w).sum() per day. Should sit at ~1.0.
      avg_net        mean of w.sum(). Should sit at ~0. If this drifts to
                     +0.3 you are running a market bet, and a good Sharpe
                     is just 2024 beta, not alpha.
      annual_turnover  sum of daily turnover / years. Expect 15-40x for a
                     daily-rebalanced stat arb book. At 5 bps that is
                     15-40 * 2 * 5bps = 1.5-4.0% a year of pure cost.
                     Your gross Sharpe has to clear that before anything.
    """
    
    df = pd.DataFrame(equity_curve).set_index('real_date')
    w = pd.DataFrame(weight_log).T              #dates x ticker
    df['return'] = df['todays_value'].pct_change()
    daily_rf = rf / TRADING_DAYS
    avg_return = df['return'].mean()
    std_return = df['return'].std()
    daily_sharpe = (avg_return - daily_rf)/ std_return
    annualized_sharpe = daily_sharpe * np.sqrt(TRADING_DAYS)
    
    ##quality check: avg_gross, and avg_net
    avg_gross = weight_log.abs().sum(axis = 1).mean()
    avg_net = weight_log.sum(axis = 1).mean()
    drawdown = {}
    v = df['return']
    dd = v/v.cummax() - 1
    drawdown['drawdown'] = dd.min()
    
    trough_date = dd.idxmin()
    peak_date = dd.loc[:trough_date].idxmax()
    
    drawdown['trough_date'] = trough_date
    drawdown['peak_date'] = peak_date
    
    return annualized_sharpe, avg_gross, avg_net, drawdown
    
    
    
    
    
    
    
    
    
    
    