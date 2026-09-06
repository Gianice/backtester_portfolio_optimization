import numpy as np
import pandas as pd


def compute_vol_ok(df: pd.DataFrame, days: int = 60, z_max: float = 1.5) -> pd.Series:
    """
    Volatility filter.

    Rolling standard deviation of returns, then a z-score of THAT against its
    own rolling mean and std. High z means volatility is unusually elevated
    relative to its own recent history.

    Returns a boolean Series: True = calm enough to take new risk.

    """
    out = df.copy()

    g = out.groupby('ticker')
    out['rolling_sd'] = g['simple_return'].transform(
        lambda x: x.rolling(window=days).std()
    )
    out['rollsd_roll_mean'] = out.groupby('ticker')['rolling_sd'].transform(
        lambda x: x.rolling(window=days).mean()
    )
    out['rollsd_roll_std'] = out.groupby('ticker')['rolling_sd'].transform(
        lambda x: x.rolling(window=days).std()
    )
    out['rollsd_z_score'] = (
        (out['rolling_sd'] - out['rollsd_roll_mean']) / out['rollsd_roll_std']
    )

    vol_ok = (out['rollsd_z_score'] < z_max) | (out['rollsd_z_score'].isna())
    return vol_ok.rename('vol_ok')


def apply_vol_filter(signals: pd.Series, vol_ok: pd.Series) -> pd.Series:
    """
    Block ENTRIES during high-volatility regimes; always allow EXITS.

    Being unable to open a new position in a storm is prudent. Being unable to
    close one is how you get hurt — so -1 always passes through untouched.
    """
    filtered = np.where(signals == 1, signals * vol_ok.astype(int), signals)
    return pd.Series(filtered, index=signals.index, name='signal')
