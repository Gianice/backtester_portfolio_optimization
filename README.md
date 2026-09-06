# Systematic Strategy Backtesting & Risk Analytics Engine

A modular backtesting engine for systematic equity strategies, built from the data layer up. The emphasis is on **making the numbers trustworthy** — realistic fills, explicit transaction costs, and no look-ahead — rather than on finding alpha.


---

## Why this exists

Most student backtests report a flattering Sharpe ratio produced by three bugs: trading at prices the strategy couldn't have known, ignoring transaction costs, and tuning parameters on the same data used to evaluate them.

This project is an attempt to build the harness that makes those mistakes hard to make — and to measure how much performance disappears once you stop making them.

---

## Data

| | |
|---|---|
| **Source** | Bloomberg terminal export (OHLCV) |
| **Universe** | Nasdaq-100 constituents |
| **Fields** | `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_LAST`, `PX_VOLUME` |
| **Period** | <!-- TODO: e.g. Jan 2015 – Dec 2025 --> |
| **Tickers after cleaning** | <!-- TODO: e.g. 100 → 87 --> |

Raw data is a wide Excel sheet with a two-level column header (ticker × field). The pipeline reshapes it to a long `MultiIndex(real_date, ticker)` frame.

---

## The data pipeline (`datalayer.py`)

Nine steps, each one addressing a specific way real market data is dirty:

1. **Load** the multi-header Excel export.
2. **Parse and validate dates** — coerce to datetime, report unparseable rows, sort chronologically, drop duplicates, set as index.
3. **Reshape wide → long** so each row is one ticker on one date.
4. **Coerce types** — force price and volume columns to numeric, and report how many values were lost to coercion. A silent string-to-NaN conversion is how bad data gets into a model.
5. **Drop non-universal tickers** — index membership changes over time. Any ticker with under 50% data coverage in any year is removed entirely, so results aren't driven by names that only existed for part of the sample.
6. **Align dates across tickers** — keep only dates where every surviving ticker reports, so cross-sectional comparisons are like-for-like.
7. **Flag impossible prices and outlier volume** — rows violating `low ≤ open, close ≤ high` are nulled; volume more than 3 standard deviations from the ticker's mean is nulled. Gaps are then forward-filled with a **limit of 3 days**, so a long outage stays visible as missing rather than being invented.
8. **Add derived series** — simple and log returns, computed per ticker.
9. **Validate** — check for residual NaNs, mismatched row counts across tickers, and negative prices. Failures are reported, not silently swallowed.

Output: `cleaned_data.csv`, indexed on `(real_date, ticker)`.

---

## Architecture

```
datalayer.py              load → clean → validate → cleaned_data.csv

strategies/
  base.py                 Strategy ABC — the contract every strategy implements
  ma_crossover.py         20/50 moving-average crossover

risk/
  filters.py              volatility-regime overlay (blocks entries, never exits)

engine/
  execution.py            fill pricing and transaction costs
  backtest.py             the backtest loop

research/
  metrics.py              performance and risk metrics

main.py                   wiring only
```

The separation is the point. `engine/backtest.py` contains no reference to moving averages; `strategies/ma_crossover.py` contains no reference to cash or fills. A new strategy is one file implementing one method, and it inherits the entire evaluation harness — same costs, same metrics, same tests — which is what makes two strategies **comparable**.

---

## Backtest conventions

Stated explicitly, because these choices are what separate a plausible backtest from a misleading one:

- **Timing.** Decide on bar *t−1*'s close → fill at bar *t*'s open → mark to market at bar *t*'s close. The execution lag is applied by the engine, not the strategy, so a strategy author cannot introduce look-ahead by accident.
- **Costs.** One `cost_bps` parameter per fill: buys fill above the reference price, sells below. It collapses half-spread, commission and market impact into a single figure — the honest resolution available from daily bars.
- **Position sizing.** Currently all-in / all-out, one position per ticker, each ticker on its own capital. This is *not* a portfolio backtest yet; a portfolio allocation layer is on the roadmap.
- **Volatility filter.** Blocks new entries when realised volatility is unusually elevated; never blocks exits. During the warm-up period, when the regime is unknown, entries are permitted — "I don't know yet" is treated differently from "conditions are dangerous."

---

## Results

<!-- TODO — fill these in once the cost sweep runs. Report real numbers, including
     the unflattering ones. A strategy that loses to buy-and-hold net of costs is a
     finding worth stating, not a failure to hide. -->

**Cost sensitivity** — the same strategy at increasing round-trip cost assumptions:

| Cost (bps) | Total return | Sharpe | Max drawdown | Round trips |
|---|---|---|---|---|
| 0 | | | | |
| 5 | | | | |
| 10 | | | | |
| 20 | | | | |

**Benchmark** — versus buy-and-hold on the same universe, net of costs: <!-- TODO -->

**Breakeven cost** — the level at which the strategy stops adding value: <!-- TODO -->

---

## Roadmap

- [x] Data pipeline with validation
- [x] MA crossover signal layer with volatility overlay
- [x] Backtest loop with realistic execution timing
- [ ] Transaction-cost model and sensitivity sweep
- [ ] Performance metrics — Sharpe, max drawdown, VaR, hit rate, turnover
- [ ] Isolation Forest anomaly detection on price and volume
- [ ] Event-driven refactor (`Portfolio` / `ExecutionHandler` over an event queue)
- [ ] Portfolio allocation layer — single capital pool, position sizing, mean-variance optimiser
- [ ] Risk attribution — marginal contribution to portfolio volatility
- [ ] Walk-forward out-of-sample harness and parameter sensitivity grid
- [ ] Second strategy (mean-reversion) to demonstrate the framework generalises

---

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl

python datalayer.py      # builds cleaned_data.csv from the Excel export
python main.py           # runs the backtest and plots an equity curve
```

---

## Notes

Built as a self-directed project to develop practical quantitative research skills — data engineering, signal construction, backtest methodology and execution-cost analysis.


by Gianice Lim Kai Qing