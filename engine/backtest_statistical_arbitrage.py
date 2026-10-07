import numpy as np
import pandas as pd


TRADING_DAYS = 252

def run_portfolio_backtest(prices, weights,
                           initial_capital=100_000.0,
                           cost_bps=0.0,
                           execution_lag=1,
                           band=0.2):                      # NEW
    """
    band : no-trade band, relative to target.
           Trade a name only if its target changed, or its actual weight
           has drifted more than band * |target| away from target.
           band=0 -> rebalance every day (old behaviour).
    """
    equity_curves = []
    weight_log = {}
    weights = (weights.reindex(index=prices.index, columns=prices.columns)
                      .shift(execution_lag).fillna(0.0))
    cash = initial_capital
    shares = pd.Series(0.0, index=prices.columns)
    prev_w = pd.Series(0.0, index=prices.columns)

    for date in prices.index:
        px = prices.loc[date]
        w = weights.loc[date]

        equity = cash + (px * shares).sum()

        # where the book actually is, before trading
        w_actual = shares * px / equity

        # rule 1: did the signal change this name's target?
        target_changed = ~np.isclose(w, prev_w)

        # rule 2: has it drifted too far from target?
        drifted = (w_actual - w).abs() > band * w.abs()

        # NEW — trade only names that hit rule 1 or rule 2
        trade = target_changed | drifted
        new_shares = shares.where(~trade, w * equity / px)

        traded = ((new_shares - shares).abs() * px).sum()
        turnover = traded / equity
        cost = traded * cost_bps / 10000

        cash = cash - ((new_shares - shares) * px).sum() - cost
        shares = new_shares
        prev_w = w                                         # NEW

        value = cash + (shares * px).sum()
        equity_curves.append({'real_date': date,
                              'todays_value': value,
                              'turnover': turnover,
                              'cost': cost,
                              'net_exposure': (shares * px).sum() / value,   # NEW
                              'n_traded': int(trade.sum())})                 # NEW
        weight_log[date] = shares * px / value

    return equity_curves, weight_log



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
    avg_gross = w.abs().sum(axis = 1).mean()
    avg_net = w.sum(axis = 1).mean()
    drawdown = {}
    v = df['todays_value']
    dd = v/v.cummax() - 1
    drawdown['drawdown'] = dd.min()
    
    trough_date = dd.idxmin()
    peak_date = v.loc[:trough_date].idxmax()
    
    drawdown['trough_date'] = trough_date
    drawdown['peak_date'] = peak_date
    
    return {'sharpe': annualized_sharpe, 'avg_gross': avg_gross,
        'avg_net': avg_net, **drawdown}
    
    
    
    
    
    
    
    
    
    
    