# Systematic Strategy Backtesting & Risk Analytics Engine

A modular backtesting engine for systematic equity strategies, built from the data layer up. The emphasis is on **making the numbers trustworthy** — realistic fills, explicit transaction costs, and no look-ahead — rather than on finding alpha.

## Two strategies

| | Strategy 1 — Moving-average crossover | Strategy 2 — PCA residual statistical arbitrage |
|---|---|---|
| **Type** | Long-only trend following, per ticker | Long/short, market- and factor-neutral portfolio |
| **Signal** | 20/50-day MA crossover with a volatility-regime filter | Mean reversion of each stock's residual after removing common factors (Avellaneda & Lee, 2010) |
| **Engine** | `engine/backtest.py` — per-ticker loop, buy/sell signals | `engine/backtest_statistical_arbitrage.py` — signed target weights, costs on turnover |
| **Status** | ✅ Complete, with results | 🚧 In progress — engine, factors, residuals and OU filter built; signal and backtest next |
| **Section** | [Results](#strategy-1--moving-average-crossover-results) | [Pipeline](#strategy-2--pca-residual-statistical-arbitrage-in-progress) |

**Strategy 1 headline:** it **captures 36% of the return available from buy-and-hold, and beats it on 13 of 87 tickers**. That is reported up front rather than buried, because the purpose of the harness is to produce numbers you can defend.

**Strategy 2** is a different kind of strategy on the same data layer: it holds long and short positions at once, so it is meant to make money whether the market rises or falls. Results will be reported against a shuffled-returns baseline, so a number only counts if it beats what the pipeline produces on pure noise.

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
| **Period** | 2 Jan 2020 – 23 Jun 2026 (1,626 trading days; the export's 1,691 rows include market holidays and a placeholder final row) |
| **Tickers after cleaning** | 101 → 88, of which 87 tradeable (`NDX Index` held out as benchmark) |
| **Rows** | 148,808 in `cleaned_data.csv` (88 × 1,691); 147,117 after `NDX Index` is held out |

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
7. **Flag impossible prices and outlier volume** — rows violating `low ≤ open, close ≤ high` are nulled; volume more than 3 standard deviations from the ticker's mean is nulled. Gaps are forward-filled with a **limit of 3 days**, so a long outage stays visible as missing rather than being invented. After the fill, 187 volume values remain missing; prices have none.
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

**8,435 bars recovered.** The residual 4.48% is still worth investigating — part of it traces to one ticker (see *Known limitations*).

The count excludes each ticker's first bar, whose return is set to zero because there is no previous close. Counting raw zeros instead gives 6,672 (4.54%) — exactly 87 more, one per ticker.

### Wide format for cross-sectional work (`data/wide.py`)

The backtest loop works ticker by ticker on the long panel; PCA needs every ticker side by side. `load_wide()` pivots the panel into two `days × tickers` frames — prices and simple returns — drops `NDX Index` (a benchmark, not a tradeable name), and removes days on which every ticker's return is exactly zero (market holidays carried through by the date alignment).

Result: **1,626 days × 87 tickers, no missing values, 2 Jan 2020 – 23 Jun 2026.** 65 all-zero days removed. Most are US market holidays, but the last is not: **24 June 2026 is a placeholder row in the vendor export**, with all 88 series carrying the previous close — every return that day is exactly zero, on a normal Wednesday. The export was most likely pulled before that day's close.

`load_wide_cached()` writes both frames to parquet and reuses them, rebuilding automatically whenever the source CSV is newer than the cache — so a fix to the cleaning pipeline can never be silently masked by stale cached data. Paths are anchored to the file's own location rather than the working directory.

---

## Anomaly detection (`data/quality.py`)

An Isolation Forest flags unusual bars on six scale-free features: return, absolute return, intraday range as a share of price, log volume, change in log volume, and the residual of absolute return regressed on log volume.

Four design choices, all deliberate:

- **Flag, don't delete.** Downstream code decides what to do. Nulling rows destroys the evidence and makes it impossible to count what was caught.
- **Fit per ticker.** A normal day for a volatile small-cap is an extreme event for a mega-cap.
- **Engineer away the diagonal.** Isolation Forest splits on one feature at a time, so its decision boundaries are axis-parallel and it approximates a diagonal relationship poorly. Large moves arrive on large volume — a diagonal in (`abs_ret`, `log_volume`) space. Regressing one on the other and handing the model the residual turns that diagonal into a single column one cut can isolate. The residual is uncorrelated with the regressor by construction.
- **`contamination` is an assumption, not a discovery.** It fixes where the threshold falls, not what the scores are.

### What it found

1,496 of 148,808 bars flagged (1.01% — by construction, not a finding; the count includes `NDX Index`, which is scored alongside the stocks). Because the model is fit per ticker on equal-length histories, every ticker is flagged the same number of times, so tickers are ranked by their single worst score instead. The count is not the result; the review is:

**Every one of the ten highest-scoring bars is a genuine event, not a data error.**

- **Seven are the March 2020 COVID crash** — EXC +18.0% on 17 March, CSCO +13.4% and PEP +10.5% on 13 March, and further moves in PEP and XEL between 12 and 20 March.
- **One is the 9 November 2020 vaccine rotation** — ROST +15.6%.
- **One is the 9 April 2025 tariff reversal** — ADI +18.4%.
- **One is stock-specific** — VRTX −20.6% on 5 August 2025.

The first nine share dates *across tickers*, which is the signature of a market-wide event. VRTX is the exception: one ticker on one day, the pattern a data error would also produce. So it was checked: the price stepped from 472 to 375 and stayed there, trading between 366 and 388 for the following week. A bad print jumps and snaps back; this is a level shift, consistent with company news that day, and is treated as genuine.

So the flag is an **extreme day** marker, not a **bad data** marker, and the cleaning pipeline passes: no fat fingers, stale prints or missed split adjustments among the worst-scoring rows. Rows are flagged, never dropped — deleting March 2020 would remove precisely the days that determine whether a strategy survives.

**The per-ticker fit shows up in who ranks highest.** By worst single day, the top of the list is utilities and defensives — EXC, XEL, PEP, PAYX — not the high-volatility names. `NVDA` does not appear. An 18% day is genuinely stranger for a utility than for a semiconductor name; a pooled fit would have ranked it the other way round.

Ranking by *mean* score inverts the list — NVDA, MU, TSLA, CRWD, STX lead — but the spread across tickers is 0.439–0.442, too narrow to mean anything. Mean score is not used.

---

## Architecture

```
data/
  datalayer.py            load → clean → validate → cleaned_data.csv
  quality.py              Isolation Forest anomaly flagging
  wide.py                 long panel → wide prices/returns, parquet cache

strategy/
  base.py                 Strategy ABC — the contract every strategy implements
  ma_crossover.py         20/50 moving-average crossover

risk/
  filters.py              volatility-regime overlay (blocks entries, never exits)

engine/
  backtest.py                         Strategy 1: per-ticker loop, fill pricing, costs
  backtest_statistical_arbitrage.py   Strategy 2: long/short portfolio, target weights, turnover costs

research/
  metrics.py              performance and risk metrics
  factors.py              Strategy 2: rolling PCA factor extraction
  residuals.py            Strategy 2: factor regression and cumulative residuals
  ou.py                   Strategy 2: OU fit and ADF stationarity filter

tests/
  test_pnl.py             hand-computed P&L assertions (Strategy 1 engine)
  test_backtest_statistical_arbitrage.py   long/short engine checks (Strategy 2 engine)
  test_pca_return_result.py                PCA sanity checks on a saved window

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

## Strategy 1 — Moving-average crossover (results)

20/50 moving-average crossover with a 60-day volatility filter, 87 tickers, Jan 2020 – Jun 2026.

**2,909 signals fired** — 1,320 entries, 1,589 exits. More exits than entries is expected: the volatility filter blocks entries but never exits. Median 30 trades per ticker over 6.5 years.

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

The index itself (8,733 → 29,347) beat the median constituent by about 100 points. That gap is cap-weighting: the index return was carried by a few very large winners, while the typical stock did much less.

This is the expected failure mode of a long-only trend-following rule during a sustained uptrend: the system sits in cash whenever the short average is below the long one, and each whipsaw exits near a local low and re-enters higher. The 2020–2026 Nasdaq-100 was close to the worst possible environment for it.

A median Sharpe of 0.333 is not an investable result. It is reported because a harness that only produces flattering numbers is not a harness.

### One row that explains the whole result

`NVDA` ranked **4th best by Sharpe (1.06)** of all 87 tickers — and returned **+786% against buy-and-hold's +3,301%**.

Both numbers are correct, and they disagree because they measure different things. Sharpe rewarded the strategy for sitting out NVDA's drawdowns; total return shows what sitting out cost — roughly 2,500 percentage points. Any evaluation that reported only risk-adjusted performance would have called this a success.

### The "wins" are mostly downside avoidance, not alpha

Of the 13 tickers where the strategy beat buy-and-hold, four are cases where **buy-and-hold lost money and the strategy lost less** (INTU, CMCSA, PYPL, KHC), and two more are names where buy-and-hold made under 16% (TRI, PAYX). Only three — WDC, META, PDD — are wins on a stock that rose strongly.

| Ticker | Strategy | Buy-and-hold | Edge |
|---|---|---|---|
| WDC | +1630% | +957% | +674 pp |
| META | +315% | +174% | +141 pp |
| PDD | +202% | +102% | +100 pp |
| INTU | +62.9% | −1.5% | +64 pp |
| TRI | +62.0% | +7.0% | +55 pp |
| CMCSA | +0.9% | −49.3% | +50 pp |
| PAYX | +60.8% | +15.2% | +46 pp |
| PYPL | −17.8% | −61.4% | +44 pp |
| KHC | −6.4% | −30.1% | +24 pp |
| CTAS | +167% | +151% | +16 pp |

And the losses are concentrated in the biggest winners:

| Ticker | Strategy | Buy-and-hold | Edge |
|---|---|---|---|
| NVDA | +786% | +3301% | −2514 pp |
| STX | +101% | +1646% | −1544 pp |
| KLAC | +127% | +1272% | −1145 pp |
| TSLA | +143% | +1268% | −1125 pp |
| CRWD | +233% | +1265% | −1033 pp |
| MU | +825% | +1856% | −1030 pp |
| LRCX | +320% | +1170% | −850 pp |
| AMD | +358% | +1034% | −675 pp |

**Four of the 15 best tickers by Sharpe — NVDA, MU, LRCX and AMD — are also among the 10 worst by edge.** NVDA is not an isolated case: on the strongest trends, a high Sharpe and a large shortfall against buy-and-hold come together.

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

**The strategy is largely cost-insensitive.** At a realistic 5 bp it loses 3% of its Sharpe, and survives to 0.224 even at 50 bp. The reason is visible in the trade count — at a median of 30 trades over 6.5 years there is very little to tax.

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

## Strategy 2 — PCA residual statistical arbitrage (in progress)

Strip out the part of each stock's move that the market and its sectors explain, and bet that what's left snaps back. Follows Avellaneda & Lee (2010), *Statistical Arbitrage in the US Equities Market*.

**The idea.** A stock's daily return is a common part (the market rose, tech outperformed) plus a stock-specific part. The common part is hedged away. The stock-specific part, summed over time, tends to drift and revert — that reversion is the edge. Chosen over pairs trading because 87 tickers give 3,741 candidate pairs to screen, each a chance for a false positive, but only 87 residuals.

### Pipeline

**A. Risk model — what to hedge** (`research/factors.py`, ✅ built)

1. **Window** — on each trade date, the previous 252 days of returns for all stocks. The trade date itself is excluded, so there is no look-ahead.
2. **Standardise** — each stock to mean 0, std 1, so volatile names don't dominate.
3. **Correlation matrix** — C (N × N).
4. **Eigendecomposition** — each eigenvector is a portfolio of weights across all stocks; each eigenvalue is how much of total variance it explains. Signs are fixed every window so a factor can't flip between days.
5. **Choose K** — keep eigenvalues above the Marchenko–Pastur noise bound (2.52 for 87 stocks on 252 days). PC1 is the market. The 55%-of-variance rule used in the original paper is computed alongside for comparison.
6. **Factor returns** — `F = Rs @ v[:, :K]`: K time series, each a common force moving day by day.

The eigenvectors `v` form an N × N matrix: rows are stocks, columns are components. A single eigenvector is one column — `v[:, 0]` is PC1's weights across all stocks — while `v[i, :]` is stock *i*'s loading on every component. The eigenvalues `w` are sorted descending and sum to N, so `w[0] / N` is the share of variance PC1 explains.

The factor returns `F = Rs @ v[:, :K]` form a window × K matrix: each column is one factor as a time series, each row one day. Shapes multiply as (252 × N) @ (N × K) → (252 × K). `F` is the same for every stock — step 7 gives each stock its own betas to it.

**How many factors? The two rules disagree, and that is the finding.**

Across 1,374 rolling windows (1,626 days less the 252-day warm-up):

| Rule | Min | Median | Max |
|---|---|---|---|
| Marchenko–Pastur noise bound | 3 | **3** | 4 |
| 55% cumulative variance (Avellaneda & Lee) | 3 | **9** | 15 |

Only 3–4 eigenvalues are ever distinguishable from noise. The variance rule asks for three times as many at the median, and swings from 3 to 15 with the correlation regime: when stocks move together, as in a crash, three factors already explain 55%; in calmer periods the variance is spread thin and the rule keeps adding components to reach the threshold. The extra factors it adds beyond the noise bound are indistinguishable from randomness — and hedging against noise strips out part of the stock-specific move the strategy is trying to trade.

K is therefore set by the Marchenko–Pastur bound. The variance rule comes from a paper run on a much larger universe; on 87 names it over-fits the risk model.

**PC1 is the market, but not a pure one.** On the saved 252-day window, PC1 explains about 15% of variance (λ₁ = 12.7 of 87) and 74 of 87 stocks load positively. The 13 that load negatively are almost all defensives — utilities (EXC, AEP, XEL), staples (PEP, MDLZ, COST, KHC, WMT) and telecom (TMUS, CMCSA) — and every one of their loadings is small: the largest is −0.064, against a median of 0.099 for the rest. In a window where defensives moved against tech, PC1 picks up a growth-versus-defensive tilt alongside the market. The test for PC1 therefore checks that wrong-signed loadings are small, not that none exist.

**B. Residuals — isolate the stock-specific move** (`research/residuals.py`, ✅ built)

7. **Regress each stock on F** — residual `ε = r − β·F`.
8. **Cumulate** — `X = ε.cumsum()`: how far the stock has drifted from where the factors say it should be.

Checks on a 252-day window with K = 3, all 87 stocks:

| Check | Result | What it confirms |
|---|---|---|
| max \|Fᵀε\| | 4.0 × 10⁻¹⁴ | Residuals are orthogonal to the factors — the regression is implemented correctly |
| max \|X on last day\| | 5.3 × 10⁻¹⁶ | X ends at exactly zero, as OLS with an intercept forces |
| Mean PC1 beta | +0.0032 | Stocks load positively on PC1, so the eigenvector sign-fixing works |
| Median residual std ÷ raw std | **0.63** | Three factors explain about 60% of a typical stock's variance (1 − 0.63²) |

The last row is the substantive one. Even after removing the market and two further common factors, **the residual still carries 63% of a typical stock's volatility.** That stock-specific part is what Strategy 2 trades — whether any of it mean-reverts is the question for step 9.

The second row has a consequence worth stating: pinning X to zero at the end of the window (and near zero at the start) makes it a path that must return, which flatters every in-window mean-reversion statistic. This is why step 9 cannot rely on the half-life alone.

**C. Mean reversion — does the drift come back?** (`research/ou.py`, ✅ built)

9. **OU fit + ADF test** — estimate κ, equilibrium level, σ and half-life from X. Trade only stocks passing ADF **and** with a 1–30 day half-life.

Result on the same 252-day window:

| | |
|---|---|
| AR(1) coefficient in (0, 1) — i.e. κ > 0 | 87 of 87 |
| Pass ADF at 5% | **11 of 87 (12.6%)** |
| Tradeable (ADF and half-life 1–30 days) | 11 of 87 |
| Median half-life, tradeable | 10.0 days |
| Positions opened at \|s\| > 1.25 | 4 (1 long, 3 short) |

**Every residual looks mean-reverting by κ, and that tells us nothing** — the Brownian-bridge effect from step 8 guarantees it. The ADF test is what actually filters, and the half-life bound removed nothing further.

**11 of 87 is not far above what noise would produce.** At a 5% significance level, about 4 of 87 would pass by chance even with no mean reversion at all — and the true chance rate here is higher than 5%, because the Brownian-bridge effect makes every residual look more stationary than it is. How much higher is not yet measured. Only two of the eleven, WDC (p = 0.0004) and TRI (p = 0.004), pass decisively; the rest sit between p = 0.02 and 0.05.

So on this window there is **no clear evidence of residual mean reversion beyond chance.** That is a result, not a failure — but it means the strategy's edge cannot be established window by window. It has to come from the ADF pass rate across all 1,374 windows compared with the same pipeline run on shuffled returns (step 13).

**D. Positions**

10. **Signal** — `s = −m / σ_eq`, a z-score of how stretched X is.
11. **Trade** — buy below −1.25, short above +1.25; close longs at |s| < 0.50, shorts at |s| < 0.75 (earlier, because shorts carry borrow cost).
12. **Portfolio** — 1× gross, checked for dollar neutrality and zero net exposure to each factor every rebalance.

**E. Validation — prove it isn't noise**

13. Costs on turnover, bootstrap CI on Sharpe, sensitivity across every parameter, and the whole pipeline re-run on **shuffled returns**. Step 9 already shows the in-window statistics flatter the data, so a result only counts if it clearly beats that baseline.

### Limitations specific to this strategy

- **Survivorship bias matters more** — PCA on today's survivors finds cleaner factors than the real historical universe would.
- **Turnover is the cost risk** — the strategy rebalances far more often than Strategy 1, so costs charged on turnover will matter much more.
- **Short borrow not modelled** — every short assumes a locate.
- **K can change between windows**, so `PC2` on one date isn't guaranteed to be the same economic factor on another.
- **Rebalancing uses a no-trade band.** A name is traded only when its target changes or its actual weight drifts more than 20% from target, so the book is dollar- and factor-neutral at the moment of trading and drifts within the band in between. `band=0` reproduces daily rebalancing exactly, and a test enforces it.

---

## Known limitations

Listed because they bound what the numbers above are worth.

1. **Survivorship bias — the most serious.** The universe is *current* Nasdaq-100 membership pulled back to 2020. Companies removed from the index between 2020 and 2026 are absent from the source export entirely. Every result is therefore computed on names successful enough to still be in the index in 2026, which inflates returns for both the strategy and the benchmark. Fixing it requires point-in-time index membership, which the current data source does not provide.
2. **In-sample only.** Parameters (20/50 windows, 60-day volatility lookback, z-threshold 1.5) were chosen by judgement rather than search, so there is no explicit overfitting — but there is also no out-of-sample evidence. A walk-forward harness is the next build.
3. **Flat transaction costs.** `cost_bps` is independent of order size. Real impact grows roughly with the square root of order size relative to average daily volume. Fine at this position scale; wrong for anything larger.
4. **Not a portfolio.** Each ticker runs independently on full capital, so the 87 results cannot be aggregated into a portfolio return, and there is no diversification, no risk budget and no correlation handling.
5. **The volume filter is destructive.** Bars with |z| > 3 on volume are nulled and forward-filled. A market panic day is by definition a >3σ volume day, so March 2020 volume is replaced by a stale value from a calm day — and `quality.py` computes its volume features from that. 187 volume values remain missing where the nulled run outlasted the 3-day fill limit. Extreme volume is information, not an error; catching the real defect (`volume <= 0`) would be better.
6. **`NBIS US Equity` is not a real price series.** It has a full 1,691-row history from 2020 with zero nulls, but 734 of those rows (43.4%) have a zero return. The company did not trade under that ticker for most of the period; the earlier rows are carried-forward placeholders in the vendor export. `drop_non_universal` misses it precisely because there are no nulls to count. It accounts for about one in nine of the 6,585 flat closes in the universe — a real contributor, but most of the residual 4.48% is spread across the other 86 tickers. The stat arb factor layer works around it by dropping zero-variance tickers per window, but the series itself is still wrong.
7. **`GOOG` and `GOOGL` are both in the universe.** They are two share classes of one company and move almost identically, so Alphabet is counted twice in every median above. For the stat arb strategy it matters more: the pair is close to perfectly correlated, which gives PCA a near-duplicate column and means one company carries double weight in the factors.
8. **Zero-return days are still in the Strategy 1 panel.** `wide.py` removes the 65 all-zero days (market holidays plus the 24 June 2026 placeholder), but `datalayer.py` does not, so the MA backtest runs on 1,691 days of which 65 have no price change. Padding a return series with zeros understates Sharpe slightly; the fix is to drop all-zero dates in `datalayer.py` so both strategies share one calendar.
9. **The anomaly model sees the full history.** Fitting on all data to judge whether a given day is unusual is look-ahead. Defensible for a one-off cleaning pass, not for anything used in a live signal.

---

## Roadmap

- [x] Data pipeline with validation
- [x] MA crossover signal layer with volatility overlay
- [x] Backtest loop with explicit execution timing
- [x] Transaction-cost model and sensitivity sweep
- [x] Performance metrics — Sharpe, max drawdown, VaR, hit rate, turnover
- [x] Isolation Forest anomaly detection on price and volume
- [x] Wide-format loader with parquet cache
- [x] PCA residual stat arb — rolling factor extraction, K by Marchenko–Pastur vs 55% variance
- [x] PCA residual stat arb — residuals and OU fit with ADF filter
- [ ] PCA residual stat arb — signal and portfolio backtest
- [ ] PCA residual stat arb — validation against a shuffled-returns null baseline
- [ ] Walk-forward out-of-sample harness and parameter sensitivity grid
- [ ] Portfolio allocation layer — single capital pool, position sizing, mean-variance optimiser
- [ ] Risk attribution — marginal contribution to portfolio volatility
- [ ] Event-driven refactor (`Portfolio` / `ExecutionHandler` over an event queue)
- [ ] Size-dependent cost model and implementation-shortfall TCA
- [ ] Point-in-time index membership to remove survivorship bias

---

## Running it

```bash
pip install -r requirements.txt

python data/datalayer.py      # builds cleaned_data.csv from the Excel export
python data/quality.py        # flags anomalous bars
python main.py                # runs the MA backtest, prints results and the cost sweep

python data/wide.py           # builds and caches the wide price/return frames
python -m research.factors    # rolling PCA; prints K by each rule across 1,374 windows

pytest tests/                 # P&L and PCA sanity checks
```

`research.factors` must be run as a module (`-m`) from the repository root so that `from data.wide import ...` resolves.

---

## Notes

Built as a self-directed project to develop practical quantitative research skills: data engineering, signal construction, backtest methodology, and execution-cost analysis.

by Gianice Lim Kai Qing