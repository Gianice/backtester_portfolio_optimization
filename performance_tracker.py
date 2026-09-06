import pandas as pd
import numpy as np
"""
performance_tracker.py
Evaluates and explains results from backtester.py and/or portfolio_optimiser.py.
Generic enough to be called by both — does not generate signals or weights itself.
"""

##all equity curve: {ticker:[{'real_date': row['real_date'], 'todays_value':todays_value}, {'real_date' : row['real_date'], 'todays_value: todays_value}]}
##all_trade_logs : {ticker:[{
                    #     'real_date' : row['real_date'],
                    #     'ticker' : row['ticker'],
                    #     'action' : 'sell',
                    #     'price': row['PX_LAST'],
                    #     'shares' : shares_held
                    # },
                    # {
                    # 'real_date' : row['real_date'],
                    #     'ticker' : row['ticker'],
                    #     'action' : 'sell',
                    #     'price': row['PX_LAST'],
                    #     'shares' : shares_held
                    # },...
                        
                # }

def calculate_returns_metrics(all_equity_curve):
    """
    Input : equity curve (backtester) or portfolio return series (optimizer)
    Output: cumulative return, CAGR, and similar performance numbers
    """
    
    """
    calculating simple return per ticker (we want to know return for each ticker and their sharpe ratio)
    """
    res = []
    for ticker in all_equity_curve.keys():
        df = pd.DataFrame(all_equity_curve[ticker])
        df = df.sort_values('real_date')
        initial_val = df.iloc[0]
        final_val = df.iloc[-1]
        returns = (final_val - initial_val)*100.0 /initial_val
    return returns



def calculate_risk_metrics(returns_series):
    """
    Input : a return series (from either system)
    Output: volatility, max drawdown (+ duration), VaR
    """
    
    pass


def calculate_sharpe_ratio(returns_series, risk_free_rate):
    """
    Input : return series, risk-free rate assumption
    Output: Sharpe ratio — shared by both backtester and optimizer evaluation
    """
    pass


def compare_to_benchmark(returns_series, benchmark_returns_series):
    """
    Input : strategy/portfolio returns, benchmark returns
            (e.g. buy-and-hold, or equal-weight portfolio)
    Output: relative performance comparison
    """
    pass


def summarize_strategy(strategy_params_or_weights, metrics_dict):
    """
    Input : the strategy logic/params or portfolio weights, plus computed metrics
    Output: a plain-language explanation of what was tested/run
            and what the results mean
    """
    pass


def plot_results(equity_curve=None, drawdown_series=None, frontier_data=None, monte_carlo_data=None):
    """
    Input : whichever result pieces are available (flexible — not all required)
    Output: visualizations — equity curve, drawdown chart,
            efficient frontier + Monte Carlo cloud, signals-on-price overlay
    """
    pass