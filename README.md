# TrustBand

**Evidence-integrity-aware financial forecasting with calibrated uncertainty.**
Perceptron Hackathon 2026 (Techfest, IIT Bombay), Theme 6: AI You Can Actually Trust, Problem 10.

> TrustBand is designed to notice when the news behind a stock forecast is duplicated, one-sided or tampered with, lean less on it, and widen or withhold its prediction instead of staying confidently wrong.

## Team InnoVerse

| Member | Role |
| --- | --- |
| Manas Girdhar (lead) | Calibration, evaluation, integration |
| Sanskriti | Data audit and TimeLock pipeline |
| Moulvi Gufran | Forecasting models and evidence-integrity signals |

## Approach

1. **TimeLock:** point-in-time alignment of news and prices at the New York close; date-only timestamps deferred to the next trading day; near-duplicates collapsed.
2. **Dual predictor:** price-only and price + news (FinBERT) LightGBM models.
3. **Evidence Jackknife:** re-forecast with each story cluster withheld; fragility = largest forecast change scaled by recent volatility.
4. **Reliance gate + calibration:** weak or fragile evidence shifts weight to the price-only model; intervals are calibrated on the final routed prediction (split conformal / CQR, then integrity-conditional conformal; conformal PID tested separately).
5. **TamperTwin:** the same archived days replayed with clean vs perturbed news, compared against baselines and ablations.

Coverage guarantees from conformal methods hold only under their stated assumptions; financial time series and tampered inputs can violate them, so all coverage is reported empirically.

## Repository

| Path | Contents |
| --- | --- |
| `PREREGISTRATION.md` | Hypotheses, systems, metrics and success criteria, fixed before the main experiments |
| `pilot.py` | Pilot runner (TimeLock, dual predictor, evidence jackknife, split conformal, tampering test) |
| `filter_fnspid.py` | Extracts a ticker/date subset from the large FNSPID news CSV |
| `results.json` | Measured pilot output |
| `test_predictions.csv` | Per ticker-day test predictions (clean, tampered, routed, fragility) |
| `pilot_run.log` | Console log of the pilot run |

## Pilot result (preliminary)

Real FNSPID data, 8 tickers (GILD, DAL, EBAY, CMG, BAC, FDX, HD, BMY), 13,255 headlines, Jan 2015 to May 2020. Test period Aug 2019 to May 2020 (1,607 ticker-days, includes the COVID-19 crash). Target interval coverage: 90%.

| System | All test days | 165 tampered days |
| --- | --- | --- |
| Price-only + split conformal | 79.4% | 72.1% |
| Price + news, clean | 79.2% | 72.7% |
| Price + news, tampered | 79.1% | 71.5% |
| Jackknife fallback, tampered | 79.3% | 72.1% |

Mean interval width was about 5.3% of price for all systems. Takeaways: all systems under-covered over a test period that includes the COVID-19 crash; daily headline features carried little signal, so tampering moved coverage by about one point and the simple fallback showed no measurable benefit. Single seed; preliminary.

### Reproduce the pilot

```bash
pip install numpy pandas scikit-learn lightgbm torch transformers
python filter_fnspid.py --input All_external.csv --out news_subset.csv --tickers GILD,DAL,EBAY,CMG,BAC,FDX,HD,BMY --start 2015-01-01 --end 2020-05-31
python pilot.py --news news_subset.csv --prices prices.csv --tickers GILD,DAL,EBAY,CMG,BAC,FDX,HD,BMY --start 2015-01-01 --end 2020-05-31 --out pilot_results
```

`prices.csv` needs columns `ticker,date,adj_close,volume`, built from FNSPID `Stock_price/full_history.zip`. Data: [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) (KDD 2024), research use only.

The pilot treats date-only and midnight timestamps as the source calendar date and makes them usable only from the next trading day; split boundaries are purged so no segment sees the next segment's closing price.

Not a trading system and not financial advice.
