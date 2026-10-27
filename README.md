# TrustBand — Reproducible Pilot Runner

**Status:** Implementation validated using **synthetic smoke-test fixtures only**. No empirical FNSPID experiment is claimed or included in this package. Never paste synthetic fixture metrics into a submission.

## What is implemented

- Ticker-day point-in-time news/price alignment (New York close; date-only/ambiguous records deferred **using original source date**; midnight-UTC-to-NY date rollover bug fixed)
- Price-only and headline+price LightGBM models
- FinBERT sentiment from **verified** `finbert_sentiment` input column or local `ProsusAI/finbert` model inference (no unlabelled proxy substituted)
- Four chronological segments: train/validation/calibration/locked test (55/15/15/15% of available *dates*) with **1-trading-session target purging** at each boundary
- Leave-one-normalized-story-cluster-out forecast sensitivity; validation-fixed 80th-percentile gating threshold
- Routed forecast to price-only for unassessable/high-fragility days
- **Separate split-conformal quantiles calibrated on original news predictor and final routed predictor**; using the standard finite-sample rank formula directly instead of requiring MAPIE
- Paired held-out duplicate-burst stress test, with actual labels/prices held fixed
- Coverage, interval width, MAE, attacked-subset counts, and fallback rate, **including the calibrated price-only baseline**
- CSV and JSON output, explicitly marked as requiring input provenance verification

## Installation

```bash
pip install numpy pandas scikit-learn lightgbm
# Only if `finbert_sentiment` has not been independently computed:
pip install torch transformers
```

The **FNSPID** repository hosts large news and price files at https://huggingface.co/datasets/Zihan1004/FNSPID . Original files are multi-GB; you may need a separate download environment. Source metadata and timestamps **must be manually audited**. FNSPID's `Date`, `Article_title`, `Stock_symbol`, `Publisher` fields are supported. Unknown-zone timestamps cannot be used same day safely.

## Prepare real data

1. Download original FNSPID `Stock_news/nasdaq_exteral_data.csv` (large). Optionally extract only relevant ticker/date rows via:

```bash
python filter_fnspid.py --input nasdaq_exteral_data.csv --out news_subset.csv --tickers AAPL,MSFT,NVDA,AMZN --start 2019-01-01 --end 2020-12-31
```

2. Provide a matching `prices.csv` with columns: `ticker,date,adj_close,volume` (split-adjusted closing prices preferred). Dates must match news range, with at least ~85 distinct usable trading dates. Extract stock-specific CSVs from FNSPID `Stock_price/full_history.zip` or prepare equivalent auditable OHLCV.
3. If original news have not been FinBERT-scored, `pilot.py` invokes `ProsusAI/finbert` via `transformers`. Requires network/cache for model weights. Else supply a `finbert_sentiment` column, signed positive-minus-negative probability (range [-1,1]), computed *with FinBERT*. Generic VADER/TextBlob scores cannot be renamed as FinBERT outputs.

Run:

```bash
python pilot.py --news news_subset.csv --prices prices.csv --tickers AAPL,MSFT,NVDA,AMZN --out pilot_results
```

Outputs: `pilot_results/results.json`, `pilot_results/test_predictions.csv`.

## What cannot be claimed

- This is **not** a complete TrustBand v2, conditional-conformal, conformal-PID, or provenance-truth-detector implementation.
- Pilot reports empirical split-conformal performance, **not guaranteed time-series coverage**.
- The heuristic story normalization is not perfect deduplication or independent-source identification.
- Attack tests measure offline sensitivity to duplicated news, **not effects on market prices**.
- Without a verified non-synthetic data run, X/Y/Z pilot numbers do not exist.
- This pilot is not financial advice, and does not implement trading.

## Audit fields reported

Observed news count, approximate normalized duplicate rate, ambiguous-time or missing-timezone fraction, publisher-field completeness, price-day coverage, sample dates, split sizes and attack-day counts. Numbers apply **only to loaded sample**, not to all FNSPID.

## Scientific caveat

The `News` set must be audited for rewritten articles, timestamp uncertainty, labels from after cutoff, survivorship and corporate actions. With only a few tickers, the empirical 90% coverage estimate is noisy and trading-day dependence invalidates standard exchangeability-based coverage claims. A pilot with no beneficial mitigation is a valid negative finding and must be reported.


## Pilot correctness updates (28 Oct 2026)

This is a **patched copy**; the earlier archive is unchanged. Fixes: preserve source calendar date for date-only/midnight entries before any NY timezone conversion; purge next-day target leakage at train/validation/calibration/test boundaries; report calibrated price-only coverage and width alongside news and routed results. Smoke data checks do not establish FNSPID empirical performance.

## Update (2026-10-28)

This runner has now been executed on real FNSPID data (8 tickers, 2015 to May 2020). Measured output is in `results/results.json` and summarised in the top-level README. The only fix needed was supplying `prices.csv` with a single price column (`adj_close`); method, splits, threshold rule, seed and attack were unchanged.
