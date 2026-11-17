# Model Card: Customer churn probability (h = 90 days)

## Purpose and use
Estimate, for each active customer on each Monday, the probability that they make **no valid purchase in the next 90 days**. The score ranks customers for a retention voucher within a fixed weekly discount budget; it does not measure whether an offer changes behaviour. Intended use is decision support for a weekly voucher list, demonstrated on historical data. Not for individual credit-like or pricing decisions outside this retailer.

## Model
- Logistic regression (L2, C = 0.01), trained on every active weekly row with per-customer weights 1 / (rows per customer), so frequent customers do not dominate.
- Features (8): recency_days, frequency_so_far, monetary_so_far, avg_order_value_so_far, tenure_days, mean_gap_so_far, recency_to_typical_gap, activity_last_90d. The skewed ones go through log1p; median imputation and standardisation follow.
- Final fit on train + validation (120,376 rows). The weights raise the effective churn rate of the training rows (62.0% weighted, 50.0% unweighted), so a log-odds offset δ = 0.259 is fitted on that refit pool to bring the mean calibrated probability back to its churn rate: `churn_probability_calibrated = expit(logit − δ)`. The test weeks are not used.
- Ten risk bands with fixed cut-offs at the validation deciles of the calibrated probability.
- Artifacts: `models/churn_model_h90.joblib` (model, preprocessing, offset, band cut-offs; `notebooks/04_modeling.ipynb`), scores in `data/results/churn_scores_h90.parquet` (`notebooks/05_scoring.ipynb`).

## Data
Online Retail II, valid purchases only (see `docs/data_dictionary.md`). Weekly snapshots from June 2010; train 2010-06-07 to 2010-11-15, validation 2011-02-14 to 2011-04-04, test 2011-07-04 to 2011-09-05, with 90-day embargoes between periods.

## Performance (test, Jul-Sep 2011; churn rate 40.1%)
| Metric | Value |
|---|---|
| AUC per week | 0.744-0.773, mean 0.759 (validation 0.780) |
| AUC, recency rule / random (pooled) | 0.638 / 0.496 |
| Brier score (calibrated) | 0.194; mean prediction 0.418 |
| HHI of the 10 risk bands | 0.110 (spread across all bands) |
| CSI vs. validation | rises from 0.08 to 0.14 (moderate shift) |
| AUC, repeat / one-time customers | 0.752 / 0.662 |

## Use in the decision
Customers are ranked by `churn_probability_calibrated` and offered a 10% voucher until the weekly expected discount cost, d·V·(p·e + (1−p)·r), reaches the budget (about £8,200, roughly 826 customers a week in the test weeks). At the same expected discount spend this list earns £9,060 of simulated net margin per weekly cohort, 49% more than ranking by days since last purchase, and leads in all ten test weeks (`notebooks/06_business_impact.ipynb`, report Chapter 7). These are scenario results, not measured campaign returns.

## Limitations
- **Seasonality.** The churn rate moves from 34.8% to 62.7% by snapshot month while the model has no calendar feature. The ranking holds, but the probability level, and with it the expected discount cost, drifts; the discount actually given in the test weeks is 1-2% above the expected cost. Recalibrate the offset quarterly (report Section 8.7).
- **One-time buyers** are ranked less accurately (AUC 0.66) and make up 38% of the voucher list but 13% of the test rows. Report pilot results separately for them.
- **Value.** Average order value is a proxy for future spend, capped at the training P99; large wholesale-like orders would otherwise dominate.
- **No causal claim.** There are no promotions in the data. Response, redemption and margin are assumptions.
- **Coverage and fairness.** Rules use behaviour only (no protected attributes), but results by country or customer size have not been checked; non-UK customers get more false positives.
- A single test season (Jul-Sep 2011).

## Monitoring
Without labels: HHI and CSI of the weekly band distribution, and the list's expected discount cost against the budget. After 90 days: AUC, Brier score, mean calibrated probability vs. observed churn, and discount given vs. expected. Recalibrate when the mean probability and the churn rate differ by more than about 5 points; retrain when AUC falls clearly below validation or CSI exceeds 0.25. If campaigns are run, measure response and redemption with a no-offer control group.
