"""
portfolio_optimiser.py
Calculates optimal asset allocation from historical return data.
Not a simulation — a one-time (or periodic) calculation.
"""

def estimate_expected_returns(price_df_multi_asset):
    """
    Input : aligned price data for multiple assets
    Output: expected (annualized) return per asset
    """
    pass


def estimate_covariance(price_df_multi_asset):
    """
    Input : aligned price data for multiple assets
    Output: covariance matrix of asset returns
    """
    pass


def generate_efficient_frontier(expected_returns, covariance_matrix, num_points):
    """
    Input : expected returns, covariance matrix, number of frontier points wanted
    Output: list/array of (risk, return) points representing the frontier
    """
    pass


def optimize_portfolio(expected_returns, covariance_matrix, objective):
    """
    Input : expected returns, covariance matrix, objective
            (e.g. 'max_sharpe' or 'min_volatility')
    Output: optimal weights (sum to 1), and that portfolio's
            resulting expected return / volatility / Sharpe ratio
    """
    pass


def run_monte_carlo(expected_returns, covariance_matrix, num_portfolios):
    """
    Input : expected returns, covariance matrix, number of random portfolios to generate
    Output: array of (risk, return, weights) for each random portfolio —
            used to visually validate the efficient frontier
    """
    pass