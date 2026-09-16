# Systematic Strategy Backtesting & Risk Analytics Engine

A modular backtesting engine for systematic equity strategies, built from the data layer up. The emphasis is on **making the numbers trustworthy** — realistic fills, explicit transaction costs, and no look-ahead — rather than on finding alpha.

The headline result is that the strategy tested here **captures 36% of the return available from buy-and-hold, and beats it on 13 of 87 tickers**. That is reported up front rather than buried, because the purpose of the harness is to produce numbers you can defend.

---

## Why this exists

Most student backtests report a flattering Sharpe ratio produced by three bugs: trading at prices the strategy couldn't have known, ignoring transaction costs, and tuning parameters on the same data used to evaluate them.

This project is an attempt to build the harness that makes those mistakes hard to make — and to measure how much performance disappears once you stop making them.

---

## Data

| | |
|---|---|
| **Source** | Bloomberg terminal export (OHLCV) |
| **Universe** | Nasdaq-100 constituents (current membership) |
| **Fields** | `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_LAST`, `PX_VOLUME` |
| **Period** | 1 Jan 2020 – 24 Jun 2026 (1,691 trading days) |
| **Tickers after cleaning** | 101 → 88, of which 87 tradeable (`NDX Index` held out as benchmark) |
| **Rows** | 147,117 |

Raw data is a wide Excel sheet with a two-level column header (ticker × field). The pipeline reshapes it to a long `MultiIndex(real_date, ticker)` frame.

---

## The data pipeline (`data/datalayer.py`)

Nine steps, each addressing a specific way real market data is dirty:

1. **Load** the multi-header Excel export.
2. **Parse and validate dates** — coerce to datetime, report unparseable rows, sort chronologically, drop duplicates, set as index.
3. **Reshape wide → long** so each row is one ticker on one date.
4. **Coerce types** — force price and volume columns to numeric, and report how many values were lost to coercion. A silent string-to-NaN conversion is how bad data gets into a model.
5. **Drop non-universal tickers** — any ticker with under 50% coverage in any year is removed. Thirteen names went this way (ABNB, ALAB, APP, ARM, CEG, CRWV, DASH, FER, GEHC, PLTR, RKLB, SNDK, WBD), all recent listings without full history.
6. **Align dates across tickers** — keep only dates where every surviving ticker reports, so cross-sectional comparisons are like-for-like.
7. **Flag impossible prices and outlier volume** — rows violating `low ≤ open, close ≤ high` are nulled; volume more than 3 standard deviations from the ticker's mean is nulled. Gaps are forward-filled with a **limit of 3 days**, so a long outage stays visible as missing rather than being invented.
8. **Add derived series** — simple and log returns, computed per ticker.
9. **Validate** — residual NaNs, mismatched row counts, negative prices, and share of flat closes. Failures are reported, not silently swallowed.

Output: `cleaned_data.csv`, indexed on `(real_date, ticker)`.

### A bug worth recording

The OHLC sanity check originally used strict inequalities (`low < open < high`). That nulls any bar where a stock **closes at its high or opens at its low** — which is common on trend days, and precisely the bars a momentum strategy cares about. Nulled bars were then forward-filled, producing a flat close and a fabricated 0.00% return.

Measured by the share of bars whose close equals the previous close:

| | Flat closes | Share |
|---|---|---|
| Strict `<` | 15,020 / 148,808 | 10.09% |
| Inclusive `<=` | 6,585 / 147,117 | **4.48%** |

**8,435 bars recovered.** Four of every five flat bars in the original dataset were manufactured by the bug. A genuinely flat close is uncommon for a liquid large-cap (1–2%), so the residual 4.48% is still worth investigating — most of it traces to one ticker (see *Known limitations*).

---

## Anomaly detection (`data/quality.py`)

An Isolation Forest flags unusual bars on six scale-free features: return, absolute return, intraday range as a share of price, log volume, change in log volume, and the residual of absolute return regressed on log volume.

Four design choices, all deliberate:

- **Flag, don't delete.** Downstream code decides what to do. Nulling rows destroys the evidence and makes it impossible to count what was caught.
- **Fit per ticker.** A normal day for a volatile small-cap is an extreme event for a mega-cap. Pooling teaches the model an average regime that fits neither.
- **Engineer away the diagonal.** Isolation Forest splits on one feature at a time, so its decision boundaries are axis-parallel and it approximates a diagonal relationship poorly. Large moves arrive on large volume — a diagonal in (`abs_ret`, `log_volume`) space. Regressing one on the other and handing the model the residual turns that diagonal into a single column one cut can isolate. The residual is uncorrelated with the regressor by construction.
- **`contamination` is an assumption, not a discovery.** It fixes where the threshold falls, not what the scores are.

### What it found

1,496 of 148,808 bars flagged (1.01% — by construction, not a finding). The count is not the result; the review is:

**Every one of the ten highest-scoring bars is a genuine market event, not a data error.** Seven are the March 2020 COVID crash; one is the 9 April 2025 tariff reversal (NXPI +21%, ADI +18%); one is the 9 November 2020 vaccine rotation (ROST +13.6%). The dates repeat *across tickers*, which is the signature of a market-wide event — a data error is idiosyncratic to one ticker on one day.

So the flag is an **extreme market day** marker, not a **bad data** marker, and the cleaning pipeline passes: no fat fingers, stale prints or missed split adjustments among the worst-scoring rows. Rows are flagged, never dropped — deleting March 2020 would remove precisely the days that determine whether a strategy survives.

One detail that shows the per-ticker fit working: `EXC` (a utility) appears three times on ±18% moves while `NVDA` barely registers. An 18% day is genuinely stranger for a utility than for a semiconductor name. A pooled fit would have missed it.

---

## Architecture

```
data/
  datalayer.py            load → clean → validate → cleaned_data.csv
  quality.py              Isolation Forest anomaly flagging

strategy/
  base.py                 Strategy ABC — the contract every strategy implements
  ma_crossover.py         20/50 moving-average crossover

risk/
  filters.py              volatility-regime overlay (blocks entries, never exits)

engine/
  backtest.py             the backtest loop, fill pricing and transaction costs

research/
  metrics.py              performance and risk metrics

tests/
  test_pnl.py             hand-computed P&L assertions

main.py                   wiring only
```

The separation is the point. `engine/backtest.py` contains no reference to moving averages; `strategy/ma_crossover.py` contains no reference to cash or fills. A new strategy is one file implementing one method, and it inherits the entire evaluation harness — same costs, same metrics, same tests — which is what makes two strategies **comparable**.

---

## Backtest conventions

Stated explicitly, because these choices are what separate a plausible backtest from a misleading one:

- **Timing.** Decide on bar *t−1*'s close → fill at bar *t*'s close → mark to market at bar *t*'s close. The execution lag is applied by the engine, not the strategy, so a strategy author cannot introduce look-ahead by accident. Filling at *t*'s **open** would be more realistic and is a planned change; filling at *t*'s close is conservative in the sense that it uses no information the signal didn't have, but it does assume a fill at a price observed at the end of the day.
- **Costs.** One `cost_bps` parameter per fill: buys fill above the reference price, sells below. A round trip therefore pays the cost twice. It collapses half-spread, commission and market impact into a single figure — the honest resolution available from daily bars.
- **Position sizing.** All-in / all-out, one position per ticker, each ticker on its own full capital. This is **not a portfolio backtest**; a portfolio allocation layer is on the roadmap.
- **Volatility filter.** Blocks new entries when realised volatility is unusually elevated; never blocks exits. During the warm-up period, when the regime is unknown, entries are permitted — "I don't know yet" is treated differently from "conditions are dangerous."
- **Correctness check.** `tests/test_pnl.py` asserts that the equity curve has one point per bar, starts at the initial capital, and that a buy-and-hold signal reproduces the raw price return to 1e-9. Every engine bug found during development was caught by a three-bar toy test, never by reading the code.

---

## Results

20/50 moving-average crossover with a 60-day volatility filter, 87 tickers, Jan 2020 – Jun 2026.

**2,909 signals fired** — 1,320 entries, 1,589 exits. More exits than entries is expected: the volatility filter blocks entries but never exits. Median 30 trades per ticker over 6.5 years, roughly 4.6 round trips a year.

### Headline: the strategy loses to doing nothing

| | |
|---|---|
| Tickers where the strategy beat buy-and-hold | **13 of 87** (15%) |
| Median total return — strategy | +45.3% |
| Median total return — buy-and-hold | **+137.1%** |
| Median edge | **−88.1 pp** |
| **Capture ratio** (Σ strategy ÷ Σ buy-and-hold) | **36.2%** |
| Median Sharpe, gross | 0.333 |
| Median Sharpe, net of 5 bp | 0.322 |
| Buy-and-hold `NDX Index` | +236.0% |

**The strategy captured just over a third of the return available from buying once and holding.** On 85% of the universe, doing nothing was better.

This is the expected failure mode of a long-only trend-following rule during a sustained uptrend: the system sits in cash whenever the short average is below the long one, and each whipsaw exits near a local low and re-enters higher. The 2020–2026 Nasdaq-100 was close to the worst possible environment for it.

A median Sharpe of 0.333 is not an investable result. It is reported because a harness that only produces flattering numbers is not a harness.

### One row that explains the whole result

`NVDA` ranked **4th best by Sharpe (1.06)** of all 87 tickers — and returned **+786% against buy-and-hold's +3,301%**.

Both numbers are correct, and they disagree because they measure different things. Sharpe rewarded the strategy for sitting out NVDA's drawdowns; total return shows what sitting out cost — roughly 2,500 percentage points. Any evaluation that reported only risk-adjusted performance would have called this a success.

### The "wins" are mostly downside avoidance, not alpha

Of the 13 tickers where the strategy beat buy-and-hold, several are cases where **buy-and-hold lost money and the strategy lost less**:

| Ticker | Strategy | Buy-and-hold | Edge |
|---|---|---|---|
| WDC | +1630% | +957% | +674 pp |
| META | +315% | +174% | +141 pp |
| PDD | +202% | +102% | +100 pp |
| INTU | +62.9% | −1.5% | +64 pp |
| CMCSA | +0.9% | −49.3% | +50 pp |
| PYPL | −17.8% | −61.4% | +44 pp |
| KHC | −6.4% | −30.1% | +24 pp |

And the losses are concentrated in the biggest winners:

| Ticker | Strategy | Buy-and-hold | Edge |
|---|---|---|---|
| NVDA | +786% | +3301% | −2514 pp |
| STX | +101% | +1646% | −1544 pp |
| KLAC | +127% | +1272% | −1145 pp |
| TSLA | +143% | +1268% | −1125 pp |
| CRWD | +233% | +1265% | −1033 pp |

The honest characterisation is therefore: **this is a drawdown-avoidance overlay, not a return generator.** On names that fell it lost less; on names that rose it captured a fraction. Whether that trade is worth making depends entirely on the objective — but it is not what "a profitable strategy" means.

### Cost sensitivity

| Cost (bps) | Median Sharpe | Median total return |
|---|---|---|
| 0 | 0.333 | +45.34% |
| 2 | 0.329 | +44.18% |
| 5 | 0.322 | +42.47% |
| 10 | 0.311 | +39.88% |
| 20 | 0.289 | +36.56% |
| 50 | 0.224 | +26.60% |

**The strategy is largely cost-insensitive.** At a realistic 5 bp it loses 3% of its Sharpe, and survives to 0.224 even at 50 bp. The reason is visible in the trade count — at ~4.6 round trips a year there is very little to tax. A strategy trading weekly would see 5 bp consume the entire edge.

This is the one genuinely favourable property the test found, and it is a property of the *turnover*, not of the signal.

### Where it worked, and why that's a warning

| Ticker | Total return | Sharpe | Max drawdown | Trades |
|---|---|---|---|---|
| WDC | +1630% | 1.32 | −50.1% | 21 |
| AVGO | +606% | 1.11 | −26.1% | 20 |
| MU | +825% | 1.06 | −42.6% | 25 |
| NVDA | +786% | 1.06 | −41.3% | 34 |
| GOOGL | +242% | 0.94 | −30.4% | 24 |

The top of the table is almost entirely **semiconductors and semiconductor equipment** — WDC, AVGO, MU, NVDA, TER, MRVL, AMAT, LRCX, AMD, ASML, LITE. Moving-average crossover works where there are long sustained trends, and that is where the trends were. This is not a general edge; it is a bet on trend persistence, and the table shows exactly which names happened to supply it.

Drawdowns of 40–60% across the leaders are worth naming too. The volatility filter blocks *entries* during turbulent regimes but has no exit rule, so it cannot protect against a drawdown that begins while a position is already open.

`WDC` tops both the Sharpe ranking and the edge ranking, which is the usual signature of a data problem — so it was checked. Its five largest daily moves are 28.7%, 20.4%, 18.3%, 17.7% and 16.8%, with no 2× or 0.5× discontinuity of the kind an unadjusted split or spinoff produces. The 10.6× price ratio is built from many moves rather than one jump, so the figure appears genuine.

---

## Known limitations

Listed because they bound what the numbers above are worth.

1. **Survivorship bias — the most serious.** The universe is *current* Nasdaq-100 membership pulled back to 2020. Companies removed from the index between 2020 and 2026 are absent from the source export entirely. Every result is therefore computed on names successful enough to still be in the index in 2026, which inflates returns for both the strategy and the benchmark. Fixing it requires point-in-time index membership, which the current data source does not provide.

2. **In-sample only.** Parameters (20/50 windows, 60-day volatility lookback, z-threshold 1.5) were chosen by judgement rather than search, so there is no explicit overfitting — but there is also no out-of-sample evidence. A walk-forward harness is the next build.

3. **Flat transaction costs.** `cost_bps` is independent of order size. Real impact grows roughly with the square root of order size relative to average daily volume. Fine at this position scale; wrong for anything larger.

4. **Not a portfolio.** Each ticker runs independently on full capital, so the 87 results cannot be aggregated into a portfolio return, and there is no diversification, no risk budget and no correlation handling.

5. **The volume filter is destructive.** Bars with |z| > 3 on volume are nulled and forward-filled. A market panic day is by definition a >3σ volume day, so March 2020 volume is replaced by a stale value from a calm day — and `quality.py` computes its volume features from that. 5.66% of volume rows carry a repeated value. Extreme volume is information, not an error; catching the real defect (`volume <= 0`) would be better.

6. **`NBIS US Equity` is not a real price series.** 1,691 rows from 2020 with zero nulls but 781 flat days (46%). The company did not trade under that ticker until late 2024; the earlier rows are carried-forward placeholders in the vendor export. `drop_non_universal` misses it precisely because there are no nulls to count. It accounts for a large share of the residual 4.48% flat-close rate.

7. **The anomaly model sees the full history.** Fitting on all data to judge whether a given day is unusual is look-ahead. Defensible for a one-off cleaning pass, not for anything used in a live signal.

---

## Roadmap

- [x] Data pipeline with validation
- [x] MA crossover signal layer with volatility overlay
- [x] Backtest loop with explicit execution timing
- [x] Transaction-cost model and sensitivity sweep
- [x] Performance metrics — Sharpe, max drawdown, VaR, hit rate, turnover
- [x] Isolation Forest anomaly detection on price and volume
- [ ] Walk-forward out-of-sample harness and parameter sensitivity grid
- [ ] Second strategy (mean-reversion) to demonstrate the framework generalises
- [ ] Portfolio allocation layer — single capital pool, position sizing, mean-variance optimiser
- [ ] Risk attribution — marginal contribution to portfolio volatility
- [ ] Event-driven refactor (`Portfolio` / `ExecutionHandler` over an event queue)
- [ ] Size-dependent cost model and implementation-shortfall TCA
- [ ] Point-in-time index membership to remove survivorship bias

---

## Running it

```bash
pip install -r requirements.txt

python data/datalayer.py    # builds cleaned_data.csv from the Excel export
python data/quality.py      # flags anomalous bars
python main.py              # runs the backtest, prints results and the cost sweep
pytest tests/               # P&L correctness assertions
```

---

## Notes

Built as a self-directed project to develop practical quantitative research skills: data engineering, signal construction, backtest methodology, and execution-cost analysis.

by Gianice Lim Kai Qing