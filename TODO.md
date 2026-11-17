# TODO

Work plan to bring the project in line with [Strategy v2](docs/strategy/README.md) and the course requirements (BA Capstone 2026, Topic 1). Each item is one branch created from the latest `main`; after a squash-merge, start the next branch from the new `main`.

Priority: **P1** required by the course · **P2** core of the strategy · **P3** robustness and report polish. Items are listed in dependency order.

## Done

- [x] MVP pipeline: Data Understanding → Evaluation, scoring output and model export (#2)
- [x] Data Preparation chapter restructured into 7 sections (#6)
- [x] `activity_last_90d` computed by key instead of row position (#7)
- [x] `latex/` renamed to `report/`; added `brief/` (2-page executive brief) and `slide/` (10-minute presentation deck) as separate top-level LaTeX deliverables
- [x] Weighted training scheme and model selection (#13); calibration offset and modeling chapter condensed to three sections (#16, #17)
- [x] Brief, slide deck and report aligned with the merged group deck "Profit-Driven Discount Strategy"; notebooks reduced to the seven that feed them (01-07)

## P1 — Required by the course

- [x] **`feat/value-and-rfm-segments`** — value proxy V and RFM segmentation (Strategy 4.7, 5.5)
  - Value concentration in notebook 01: revenue share of the top 1% / 10% of customers
  - `value_v` = average order value so far, winsorized at P99 fitted on the training period
  - RFM quintiles fitted on the training period, mapped to named segments (Champions, Loyal, Potential, At Risk, Can't Lose, Hibernating); per-segment count, revenue share, churn rate, average V
  - New columns in `modeling_dataset_h90.parquet`; update Chapters 4–5
  - Done when: segment table and V cap are in the report, cut-offs come from the training period only

- [x] **`feat/policy-simulation`** — promotion scenarios and policy comparison (Strategy 7.6–7.10)
  - Done as an equal-budget comparison (#18, notebook 06): risk ranking vs expected margin vs recency vs random at the same expected discount spend, Light/Base/Deep vouchers and m ∈ {30, 40, 50%}. The equal-contact P0–P6 grid over K was dropped because no deliverable uses it
  - E[ΔM] per customer; realized margin M_policy on test with actual labels
  - Scenarios Light (5%, 20%), Base (10%, 30%), Deep (20%, 40%) × K ∈ {10, 20, 30, 40, 50%}; sensitivity m ∈ {30, 40, 50%}, r ∈ {20, 40, 60%}
  - Policies P0 No offer, P1 Blanket, P2 Random, P3 Recency, P4 RFM-segment, P5 Model risk-only, P6 Model value-aware
  - Replaces the current e-only scenario tables (Chapter 7.6–7.10) and their assumptions (m = 35%, r = 20%, flat £5 incentive)
  - Done when: policy table at the base scenario and operating K, full grid in an appendix, value captured@K reported

- [x] **`docs/deliverables`** — required project package (Strategy 3.7, 9.4) (#18)
  - `README.md`: setup, run order of notebooks 01 → 07, outputs of each step
  - `docs/data_dictionary.md` + source log (URL, download date, license, filters, joins, cleaning log)
  - `docs/model_card.md` (one page)
  - Executive brief: done separately as `brief/main.tex` (not `docs/executive_brief.md`)
  - Done when: a third party can rerun the pipeline from the README

- [x] **`feat/dashboard`** — 3–5 KPIs and one actionable view (Strategy 8.5); built from scratch (#18, #20)
  - KPIs: active customers, revenue at risk in the target list, expected incremental margin, share of offers to likely buyers, customers per action
  - Controls for K, discount scenario and margin; target list with actions; policy comparison chart
  - Done when: it runs from the score table without re-training

## P2 — Core of the strategy

- [x] **`feat/calibration-check`** — Platt / isotonic vs raw LR, fitted on validation (Strategy 6.6); keep only if Brier improves without hurting ranking
  - Kept the intercept offset fitted on the refit pool (report Section 6.3); recalibration is part of monitoring (Section 8.7)

- [x] **`feat/value-aware-targeting`** — recommendation layer and full score table (Strategy 8.3–8.4)
  - Actions Voucher / No voucher by the weekly budget rule (notebook 07); Retain / Nurture / No-offer was replaced
  - Score table columns: `value_v`, `rfm_segment`, `expected_margin`, `rank`, `action`, `model_version`
  - Depends on: value-and-rfm-segments, policy-simulation

- [ ] **`chore/config`** — single `config.yaml` for H, R_max, C, warm-up, embargo, K, (d, e), m, r, contact cost, value cap (Strategy 2.10); notebooks read from it

## P3 — Robustness and report

- [ ] ~~**`feat/rolling-backtest`**~~ — built in #18 (notebook 11), removed because no deliverable uses it; still the strongest robustness check to add back if the report needs it (Strategy 7.5)
- [ ] **`feat/robustness-checks`** — with/without V cap, sensitivity grid, optional suppression run and H = 60 (Strategy 9.1); voucher and margin scenarios are done (report Section 7.5)
- [x] **`docs/report-structure`** — align the LaTeX report with Strategy v2
  - Chapter 8 → Targeting Policy & Decision Support (merge Feedback/monitoring into 8.7)
  - Chapter 9 → Robustness, Ethics & Limitations (ethics and bias section is new)
  - Chapter 10 → Conclusion & Recommendations with the recommended policy, K and discount
  - Chapters 1–2: business decision, unit of analysis, KPIs, success criteria v2, configuration table
