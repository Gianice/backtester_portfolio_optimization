"""
data/quality.py — multivariate anomaly flagging for OHLCV bars.

Adds two columns to the cleaned frame:
    is_anomaly     bool   True if flagged
    anomaly_score  float  higher = more anomalous (comparable within a ticker)

Design decisions, all deliberate:

  1. FLAG, DON'T DELETE.  Downstream code decides what to do. Nulling rows
     destroys the evidence and makes it impossible to count what was caught.

  2. FIT PER TICKER. Every stock has its own idea of a "normal" day. 
#     If you fit one model across both, it learns
#     an average that fits neither — it flags all the biotech's ordinary days
#     and misses the utility's genuinely odd ones.

  3. FEATURES ARE SCALE-FREE.  Returns, ratios and log-volume are comparable
     across tickers and across a decade. Raw prices are not.

  4. FITTING ON FULL HISTORY IS LOOK-AHEAD.  To judge whether today is odd,
     the model has seen next year. Defensible for a one-off cleaning pass —
     but say so in the README rather than leaving it unmentioned.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Columns the model sees. Order does not matter; the tree picks at random.
FEATURES = ['ret', 'abs_ret', 'range_pct', 'log_volume', 'vol_change',
            'move_vs_volume']

MIN_OBS = 100          # below this, a per-ticker fit is not meaningful


# ---------------------------------------------------------------------------
# 1. features
# ---------------------------------------------------------------------------

def _build_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Features for ONE ticker, already sorted by date.

    Note there is no standardisation step. Isolation Forest is scale-invariant
    per feature: a split is drawn uniformly between that feature's own min and
    max inside the current node, so multiplying a column by 1000 changes
    nothing about which partitions are likely.
    """
    f = pd.DataFrame(index=g.index)

    f['ret'] = g['simple_return']
    f['abs_ret'] = g['simple_return'].abs()

    # the day's trading range, scaled by price so it compares across tickers
    f['range_pct'] = (g['PX_HIGH'] - g['PX_LOW']) / g['PX_LAST']


    # log volume: volume is heavily right-skewed, logs make it roughly normal
    f['log_volume'] = np.log(g['PX_VOLUME'].replace(0, np.nan))
    f['vol_change'] = f['log_volume'].diff()

    # ---- the feature that fixes the axis-parallel problem -----------------
    # Big moves usually come on big volume. That relationship is DIAGONAL(slope != 0, meaning corrrelate) in
    # (abs_ret, log_volume) space, and axis-parallel splits approximate a
    # diagonal poorly. So compute the relationship ourselves and hand the
    # model the residual: "how much bigger or smaller was today's move than
    # this volume would normally imply?"  Now it is a single column, and one
    # cut can isolate it.
    
    ok = f[['abs_ret', 'log_volume']].notna().all(axis=1)
    if ok.sum() > 30:
        slope, intercept = np.polyfit(f.loc[ok, 'log_volume'],
                                      f.loc[ok, 'abs_ret'], 1)
        expected = intercept + slope * f['log_volume']
        f['move_vs_volume'] = f['abs_ret'] - expected 
        ##residual of abs_ret that is not explained by log_volume
    else:
        f['move_vs_volume'] = np.nan

    return f.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 2. the model
# ---------------------------------------------------------------------------

def flag_anomalies(df: pd.DataFrame,
                   contamination: float = 0.01,
                   n_estimators: int = 300,
                   random_state: int = 42, ##42 no meaning at all, just by default
                   min_obs: int = MIN_OBS) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : long frame indexed by (real_date, ticker), with PX_* and simple_return
    contamination : assume 1% of rows are bad.
    n_estimators : how many trees in the forest
    random_state : the seed -> starting number you feed that formula so it produce same result everytime
    min_obs : the minimum number of point for it to work, or else skip

    Returns
    -------
    A copy of df with `is_anomaly` and `anomaly_score` added.
    """
    out = df.copy()
    out['is_anomaly'] = False
    out['anomaly_score'] = np.nan

    for ticker, g in out.groupby(level='ticker', sort=False):
        feats = _build_features(g)

        # A tree cannot split on a missing value, so rows with any NaN feature
        # are excluded from the fit. They stay in the frame, unflagged.
        usable = feats.dropna()
        if len(usable) < min_obs:
            continue

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=256,       # the paper's default; bigger is usually worse
            random_state=random_state,
            n_jobs=-1, #how many CPU cores to build trees on, -1 : use every core you got, 1: single core, 2: 2 core....
            # trees are independent so just use every core you got
        ).fit(usable[FEATURES])

        # predict(): -1 = anomaly, +1 = normal
        flags = model.predict(usable[FEATURES]) == -1 #return an array

        # score_samples() returns the NEGATIVE of the textbook score, so
        # higher means MORE NORMAL. Negate it so higher = more anomalous,
        # which is what anyone reading the column will expect.
        scores = -model.score_samples(usable[FEATURES])

        out.loc[usable.index, 'is_anomaly'] = flags
        out.loc[usable.index, 'anomaly_score'] = scores

    return out


# ---------------------------------------------------------------------------
# 3. reporting — you must look at what was caught
# ---------------------------------------------------------------------------

def anomaly_report(df: pd.DataFrame, top_n: int = 10) -> None:
    """Print what the model flagged. Never trust the count alone."""
    n_flag = int(df['is_anomaly'].sum())
    print(f'flagged {n_flag} of {len(df)} rows  ({n_flag/len(df)*100:.2f}%)')
    print()

    print('most extreme tickers (by worst single day):')
    print(df.groupby(level='ticker')['anomaly_score'].max().nlargest(8).to_string())
    print()
    print('most consistently odd (by mean score):')
    print(df.groupby(level='ticker')['anomaly_score'].mean().nlargest(8).to_string())



    cols = [c for c in ['PX_LAST', 'PX_VOLUME', 'simple_return',
                        'anomaly_score'] if c in df.columns]
    worst = df[df['is_anomaly']].nlargest(top_n, 'anomaly_score')[cols]
    print(f'{top_n} most anomalous rows:')
    print(worst.to_string())


def plot_anomalies(df: pd.DataFrame, ticker: str, path: str = None):
    """Eyeball one ticker. If the flags are not obviously strange, the
    features are wrong — changing `contamination` will not help."""
    import matplotlib.pyplot as plt

    g = df.xs(ticker, level='ticker')
    flag = g['is_anomaly'].fillna(False)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    ax[0].plot(g.index, g['PX_LAST'], lw=.9, color='#1F2A44')
    ax[0].scatter(g.index[flag], g.loc[flag, 'PX_LAST'], s=34, color='#C0392B',
                  zorder=5, marker='D')
    ax[0].set_ylabel('price'); ax[0].set_title(f'{ticker} — flagged bars')

    ax[1].bar(g.index, g['PX_VOLUME'], width=1.0, color='#9AA5B1')
    ax[1].bar(g.index[flag], g.loc[flag, 'PX_VOLUME'], width=2.0,
              color='#C0392B')
    ax[1].set_ylabel('volume')

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    df = pd.read_csv('cleaned_data.csv',
                     index_col=['real_date', 'ticker'], parse_dates=['real_date'])

    flagged = flag_anomalies(df, contamination=0.01)
    anomaly_report(flagged)
    plot_anomalies(flagged,'NVDA US Equity')

    # flagged.to_csv('cleaned_data_flagged.csv')