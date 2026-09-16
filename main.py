import pandas as pd

from strategy.ma_crossover import MACrossover
from risk.filters import compute_vol_ok, apply_vol_filter
from engine.backtest import run_backtest

BENCHMARK = 'NDX Index'
COST_LEVELS = [0, 2, 5, 10, 20, 50]

def load():
    df = pd.read_csv('cleaned_data.csv',
                     index_col=['real_date', 'ticker'],
                     parse_dates=['real_date'])
    # the index itself is not a tradeable name — keep it aside as a benchmark
    is_bm = df.index.get_level_values('ticker') == BENCHMARK
    return df[~is_bm], df[is_bm].xs(BENCHMARK, level='ticker')


def diagnostics(df):
    flat = df.groupby(level='ticker')['PX_LAST'].apply(
        lambda s: (s.diff() == 0).sum()).sum()
    print(f'rows            {len(df):,}')
    print(f'tickers         {df.index.get_level_values("ticker").nunique()}')
    print(f'flat closes     {flat:,}  ({flat/len(df)*100:.2f}%)   '
          f'[pre-fix baseline was 15,020 / 10.09%]')
    print()
    

def run_all(df, signals, cost_bps=0.0):
    rows = {}
    for ticker in df.index.get_level_values('ticker').unique():
        g = df.xs(ticker, level='ticker')
        s = signals.xs(ticker, level='ticker')
        try:
            _, _, stats = run_backtest(g, s, cost_bps=cost_bps)
            rows[ticker] = stats
        except Exception as e:
            print(f'  skipped {ticker}: {type(e).__name__}: {e}')
    return pd.DataFrame(rows).T



if __name__ == '__main__':
    df, bench = load()
    diagnostics(df)

    strat = MACrossover(20, 50)
    signals = apply_vol_filter(strat.generate_signals(df), compute_vol_ok(df))
    print(f'signals fired   {int((signals != 0).sum()):,}  '
          f'({int((signals == 1).sum()):,} entries, '
          f'{int((signals == -1).sum()):,} exits)')
    print()

    res = run_all(df, signals, cost_bps=0.0)
    print('top 15 by Sharpe (gross):')
    print(res[['total_return', 'sharpe_ratio', 'depth', 'n_trades']]
          .sort_values('sharpe_ratio', ascending=False).head(15).to_string())
    print()
    print(f'median Sharpe {res["sharpe_ratio"].median():.3f}   '
          f'median trades {res["n_trades"].median():.0f}   '
          f'tickers {len(res)}')
    print()

    bh = df.groupby(level='ticker')['PX_LAST'].apply(lambda s: s.iloc[-1]/s.iloc[0] - 1)
    cmp = pd.DataFrame({'strategy': res['total_return'], 'buy_hold': bh})
    cmp['edge'] = cmp['strategy'] - cmp['buy_hold']
    print(f'strategy beat buy-and-hold on {int((cmp["edge"]>0).sum())} of {len(cmp)} tickers')
    print(cmp.sort_values('edge', ascending=False).head(10).to_string())
    print(cmp.sort_values('edge').head(10).to_string())

    print('cost sensitivity:')
    for bps in COST_LEVELS:
        r = run_all(df, signals, cost_bps=bps)
        print(f'  {bps:>3} bp   median Sharpe {r["sharpe_ratio"].median():>7.3f}   '
              f'median return {r["total_return"].median()*100:>8.2f}%')
        
    w = df.xs('WDC US Equity', level='ticker')['PX_LAST']
    print(w.pct_change().abs().nlargest(5).to_string())
    print(w.iloc[0], '->', w.iloc[-1])
    
    
    print(f'median buy-and-hold   {cmp["buy_hold"].median()*100:+.1f}%')
    print(f'median edge           {cmp["edge"].median()*100:+.1f}%')
    print(f'capture ratio         {cmp["strategy"].sum()/cmp["buy_hold"].sum():.1%}')