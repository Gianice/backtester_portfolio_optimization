"""engine/backtest.py — the per-ticker simulation loop."""
import pandas as pd

BUY, SELL = 1, -1

# ---------------------------------------------------------------------------
# What itertuples() returns
# ---------------------------------------------------------------------------
#
#   df:
#                   PX_LAST  shares  my col
#       2020-01-02    100.5    1000       a
#       2020-01-03    101.0    2000       b
#
# df.itertuples() gives an ITERATOR — rows are produced one at a time, not
# built into a list up front. Each row is a namedtuple named 'Pandas':
#
#       Pandas(Index=Timestamp('2020-01-02'), PX_LAST=100.5, shares=1000, _3='a')
#       Pandas(Index=Timestamp('2020-01-03'), PX_LAST=101.0, shares=2000, _3='b')
#
# Fields:
#   row.Index     the index value  (capital I — it is not row.index)
#   row.PX_LAST   by attribute, not row['PX_LAST']
#   row._3        'my col' has a space, so it is renamed to its position
#
# Dtypes survive: shares stays int 1000, not 1000.0 as iterrows would give,
# because a namedtuple holds separate objects rather than one typed array.
# ---------------------------------------------------------------------------


###Format of df_ticker: 
#   df:
#                   PX_LAST  shares .....??? check!!!
#       2020-01-02    100.5    1000      
#       2020-01-03    101.0    2000      
#
def apply_costs(price : float, side: int, cost_bps: float = 0.0) -> float:
    """side: + 1 = buy, -1 = sell"""
    return price * (1 + side * cost_bps / 10000)

def run_backtest(df_ticker : pd.DataFrame, 
                 signals: pd.Series,
                 initial_capital: float = 100_000.0,
                 cost_bps : float = 0.0,
                 execution_lag : int = 1):
    
    """
    One ticker, chronological order, no look-ahead.

    Decide on bar t-1's close -> fill on bar t -> mark to market on bar t's close.
    The lag is applied HERE, not in the strategy, so the strategy stays a pure
    function of prices and the engine owns all timing.

    Returns (equity_curve, trade_log, stats).
    """
    
    exec_signals = signals.reindex(df_ticker.index).shift(execution_lag).fillna(0)
    
    work = df_ticker.join(exec_signals.rename('sig'))
    
    cash = initial_capital
    shares = 0.00
    equity_curve, trade_log = [], []
    ##equity_curve format: [{'real_date': real_date, 
    #                        'todays_value': value}, 
    #                        ....]
    
    ##trade_log format: [{'real_date' : real_date,
    #                     'signal': 1/-1/0,
    #                     'shares' : ,
    #                     'balanced_cash': ,
    #                     'share_price': xx},
    #                    {......}]
    
    ##equity curve record daily value of what you own,
    ##trade log record value you trade
    for row in work.itertuples():
        exec_signals = row.sig
        if exec_signals == 1 and shares == 0.00:
            fill = apply_costs(row.PX_LAST, 1, cost_bps)
            shares = cash / fill
            date = row.Index
            cash = 0       
            trade_log.append({'real_date': row.Index,
                              'signal': exec_signals,
                              'shares': shares,
                              'price': fill,
                              'cash': cash
                              })
        elif exec_signals == -1 and shares > 0.00:
            fill = apply_costs(row.PX_LAST, -1, cost_bps)
            cash = shares * fill
            date = row.Index
            shares = 0
            trade_log.append({'real_date': row.Index,
                                'signal': exec_signals,
                                'shares': shares,
                                'price': fill,
                                'cash': cash})
        equity_curve.append({'real_date': row.Index, 'todays_value': cash + shares * row.PX_LAST})
    
    stats = compute_stats(equity_curve, trade_log)
    
    return equity_curve, trade_log, stats
        
    
def compute_stats(individual_equity_curve, trade_log):
    from research.metrics import to_returns, total_return, sharpe, max_drawdown
    
    returns = to_returns(individual_equity_curve)
    
    total_returns = total_return(individual_equity_curve)
    
    sharpe_r = sharpe(returns, rf = 0.01)
    
    depth, peak_date, trough_date = max_drawdown(individual_equity_curve)
    
    return {
        'total_return' : total_returns,
        'sharpe_ratio' : sharpe_r,
        'total_return' : total_returns,
        'depth' : depth,
        'peak_date' : peak_date,
        'trough_date' : trough_date,
        'n_trades' : len(trade_log),
        'n_bars' : len(individual_equity_curve)
        
    }
    
    
    
    
    
    
    
            
        
        
    
    
    
    
    


