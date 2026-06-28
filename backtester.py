"""
backtester.py
Simulates one trading strategy on historical price data.
Depends on: datalayer.py output. Does not calculate Sharpe/drawdown itself.
"""


def generate_signals(price_df, strategy_params):
    """
    Input : cleaned price data (one asset), strategy settings
    Output: price_df + 'signal' column (buy/sell/hold)
            Must only use data available up to each day — no look-ahead.
    """
    pass


def run_backtest(price_df_with_signals, starting_cash, cost_per_trade):
    """
    Input : signals, starting capital, trading cost assumptions
    Output: equity_curve (daily portfolio value), trade_log (per-trade detail)
    """
    pass


def summarize_backtest(equity_curve, trade_log, strategy_params):
    """
    Input : results from run_backtest()
    Output: packaged result object/dict — raw results only,
            no performance/risk calculations (hand off to performance_tracker.py)
    """
    pass


def backtest_strategy(price_df, strategy_params, starting_cash, cost_per_trade):
    """
    Orchestration only: generate_signals -> run_backtest -> summarize_backtest
    Build/test the three functions above individually before wiring this.
    """
    pass