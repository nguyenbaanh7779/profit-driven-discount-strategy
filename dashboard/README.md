# Retention targeting dashboard

Run from the repo root, after notebooks 05-07 and `pip install -r requirements.txt`:

    python src/make_rfm8.py
    streamlit run dashboard/app.py

Needs `data/results/customer_scores_actions_h90.parquet` (notebook 07), `data/results/rfm8_segments.parquet` (`src/make_rfm8.py`) and `data/processed/modeling_dataset_h90.parquet`. No model is retrained.

**KPIs (current list, scored Monday 5 Dec 2011):** active customers; customers who get a voucher; expected discount cost; revenue at risk in the list; expected extra profit over the next 90 days.

**Actionable view:** the sidebar sets the weekly discount budget (default £8,220, the budget of report Section 7.3) and the offer scenario. The page then shows who is likely to stop buying (eight behaviour groups), who gets a voucher (highest churn risk until the budget is used), a replay on the Jul-Sep 2011 test weeks against the longest-inactive rule and an offer to everyone, and the downloadable voucher list.
