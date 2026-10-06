"""Turn the MultiIndex panel into the wide frames to fulfill PCA residual statistical arbitrage."""

import os
from pathlib import Path

import pandas as pd

BENCHMARK = 'NDX Index'

ROOT = Path(__file__).resolve().parent.parent          # data/wide.py -> repo root
DEFAULT_CSV = ROOT / "cleaned_data_flagged.csv"
DEFAULT_CACHE = ROOT / "data" / "cache"


def load_wide(csv_path=DEFAULT_CSV, drop_dead=True):
    """
    Returns (prices, returns), both (days x tickers).

    1. Drop NDX Index as it is BENCHMARK and cannot be traded
    2. drop_dead : remove days where returns across all tickers are zero (very UNUSUAL -> public holiday)
    """
    df = pd.read_csv(csv_path, parse_dates=['real_date'])
    df = df.set_index(['real_date', 'ticker']).sort_index()

    prices = df['PX_LAST'].unstack('ticker').drop(columns=[BENCHMARK])
    returns = df['simple_return'].unstack('ticker').drop(columns=[BENCHMARK])

    if drop_dead:
        alive = returns.abs().sum(axis=1) > 0
        returns = returns[alive]
        prices = prices.loc[returns.index]

    return prices, returns


def load_wide_cached(source_path=DEFAULT_CSV, cache_dir=DEFAULT_CACHE):
    """
    Same output as load_wide, cached to parquet.
    Cache rebuilds automatically if the source CSV is newer than the cache.
    """
    cache_dir = Path(cache_dir)
    prices_path = cache_dir / "wide_prices.parquet"
    returns_path = cache_dir / "wide_returns.parquet"

    source_mtime = os.path.getmtime(source_path)
    cache_fresh = (
        prices_path.exists()
        and returns_path.exists()
        and os.path.getmtime(prices_path) > source_mtime
    )

    if cache_fresh:
        return pd.read_parquet(prices_path), pd.read_parquet(returns_path)

    prices, returns = load_wide(source_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)
    return prices, returns


if __name__ == "__main__":
    prices, returns = load_wide_cached()
    print("shape:", returns.shape)
    print("dates:", returns.index.min().date(), "to", returns.index.max().date())
    nan_counts = returns.isna().sum()
    print("tickers with NaN:", (nan_counts > 0).sum())
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))