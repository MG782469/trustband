# TrustBand pre-registration

Committed on 2026-10-28, before the main experiments. A small pilot (results in `pilot/results/`) was run on 2015 to May 2020 data; the main study's locked test period (2023) has not been examined.

## Primary hypothesis

**H1.** Under controlled news tampering, full TrustBand shows a smaller drop in empirical 90% interval coverage than the starter-style baseline (price + FinBERT + split conformal), at a mean interval width no more than 10% larger and with prediction availability of at least 80%.

## Secondary hypotheses

- **H2.** Integrity-conditional calibration reduces the coverage gap between low- and high-integrity days compared with a single pooled calibration.
- **H3.** Evidence-jackknife fragility detects tampered ticker-days better than chance (AUROC > 0.5 with a 95% CI excluding 0.5), and does not rise on negative controls (formatting changes, meaning-preserving paraphrases).
- **H4.** Online correction (conformal PID) reduces rolling coverage shortfall after regime breaks compared with static calibration.

## Data

FNSPID news and prices; 10 to 30 liquid tickers selected by news coverage after a timestamp audit. Chronological splits: train 2018 to 2020, model selection 2021, calibration 2022, locked test 2023. Splits may be shifted only for data-coverage reasons documented before any test-period evaluation.

## Systems compared

| ID | System |
| --- | --- |
| B0 | Price-only + split conformal |
| B1 | Price + FinBERT + split conformal (starter baseline) |
| B2 | B1 + adaptive conformal (MAPIE ACI/EnbPI) |
| B3 | B1 + integrity-conditional calibration |
| B4 | B3 + evidence-jackknife fragility |
| B5 | Full TrustBand (reliance gate, fallback, calibration on the routed prediction) |

Ablations remove each integrity signal, the jackknife and the fallback in turn.

## Tampering suite (TamperTwin)

Duplicate-story bursts, one-word polarity flips, contradictory stories, source removal, missing metadata; three severity levels; at least 5 random seeds. Historical prices are held fixed.

## Metrics

- Primary: empirical coverage at 90% on tampered days, and its change from clean.
- Interval quality: mean width, Winkler score.
- Conditional: worst-group coverage by integrity level and volatility regime; rolling 60-day coverage.
- Selective prediction: error at matched abstention rates; width at matched coverage.
- Point accuracy: MAE. Cost: runtime.
- Uncertainty: date-block bootstrap 95% confidence intervals.

## Rules

- Thresholds, features and model choices are fixed on the validation period; the test period is evaluated once per system.
- All results are reported, including null or negative ones.
- Any deviation from this plan is listed with its reason in the final report.
