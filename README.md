# backtester_portfolio_optimization

## 1.Data Cleaning — `datalayer.py`

Cleans and validates historical OHLCV market data before it's used by the backtester or portfolio optimizer. This module is responsible for producing **trustworthy data** — it does not generate trading signals, compute risk metrics, or make any strategy-related decisions.

### Pipeline order

```
load_raw_data
  → clean_data_date
  → change_to_long
  → check_type
  → drop_non_universal
  → check_misalign_date
  → check_price_and_volume
  → add_return_columns
  → validate_data
```

Each step assumes the previous one has already run — the order is deliberate (e.g. type mismatches and universe filtering are resolved before price/volume anomaly checks, since those checks need clean numeric data to be meaningful).

### What each function does

| Function | Purpose |
|---|---|
| `load_raw_data` | Reads the raw Bloomberg-style Excel export (two-row header) into a DataFrame. |
| `clean_data_date` | Parses dates (handling mixed formats via `errors='coerce'`), sorts chronologically, drops duplicate dates, sets date as index. |
| `change_to_long` | Reshapes from wide (ticker as column level) to long format (ticker as row index) via `.stack(level=0)`, for row-based filtering. |
| `check_type` | Forces OHLCV columns to numeric via `pd.to_numeric(errors='coerce')`, catching any stray non-numeric values from the source file. |
| `drop_non_universal` | Drops any ticker where the proportion of missing data in any OHLCV field exceeds 50% — treated as evidence the ticker wasn't part of the tradable universe for that period. |
| `check_misalign_date` | Keeps only dates where every ticker has a row, ensuring all assets share an identical, fully-aligned date range (required for the optimizer's covariance calculation). |
| `check_price_and_volume` | Nulls OHLC fields where High/Low/Open/Close relationships are inconsistent; nulls `PX_VOLUME` where its Z-score (per ticker) exceeds 3. Forward-fills (limit=3) afterward. |
| `add_return_columns` | Adds `simple_return` and `log_return` (both computed per-ticker via `groupby`). First-date NaNs (no prior price to compare against) are filled with 0. |
| `validate_data` | Final pass/fail check: leftover NaNs, mismatched row counts across tickers, negative prices. Reports issues without halting the pipeline. |

### Key design decisions

- **Forward-fill, not interpolation or dropping**, for short gaps (≤3 consecutive days) — avoids inventing values while tolerating brief data feed hiccups.
- **Drop the entire ticker**, not just the bad year, when historical coverage is too sparse — keeps every remaining ticker's date range fully aligned, which is required for the optimizer and avoids the complexity of a "dynamic universe."
- **Price and volume anomaly checks are independent** — a bad price doesn't automatically null volume, and vice versa, since the two are measuring different things and can fail independently.
- **Z-score (not a fixed mean/threshold) for volume outliers** — accounts for each ticker's own natural volatility rather than applying one flat cutoff to all tickers equally. Known limitation: a single extreme outlier can inflate the mean/std it's being compared against, understating its own deviation. Median/MAD or Isolation Forest are documented as possible future upgrades, not implemented in v1.
- **Wide format preserved as the final output shape** — cleaning logic is done in long format (easier row-based filtering/grouping), but reshaped back to wide before handoff, since the portfolio optimizer's covariance calculation requires tickers as columns.

### Known limitations (deliberately scoped out of v1)

- **Stock split / dividend adjustment is not implemented.** Whether the source data provides adjusted close is unconfirmed; this is on hold pending clarification. A split during the dataset's date range would currently appear as a large, uncorrected price drop.
- **Statistical outlier detection is rule-based only** (OHLC consistency + volume Z-score), not model-based. A more robust version (e.g. Isolation Forest across multiple features simultaneously) is a planned future extension, not part of v1.
- **`validate_data` reports leftover NaNs without distinguishing "expected" (gaps longer than the ffill limit) from "unexpected."** Currently relies on manual inspection of the printed output rather than automated gap-length classification.

### Output

A wide-format DataFrame — dates as rows, `(ticker, field)` as columns — with clean OHLCV data, return columns added, ready for use by `backtester.py` and `portfolio_optimiser.py`.