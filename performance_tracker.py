"""
performance_tracker.py
Evaluates and explains results from backtester.py and/or portfolio_optimiser.py.
Generic enough to be called by both — does not generate signals or weights itself.
"""

def calculate_returns_metrics(equity_curve_or_portfolio_returns):
    """
    Input : equity curve (backtester) or portfolio return series (optimizer)
    Output: cumulative return, CAGR, and similar performance numbers
    """
    pass


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