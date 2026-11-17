"""Generate notebooks/07_action_table.ipynb (run from the repo root: python src/build_nb07.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 8. Customer Action Table (Deliverable)

Report Sections 8.3-8.4. Takes the score table of `05_scoring.ipynb`, adds customer value
`value_v`, the RFM segment and the customer type, applies the weekly voucher rule of the
brief and produces:

1. the **customer-level score table with actions** (Voucher / No voucher) for every scored
   week, which `dashboard/app.py` and `src/make_rfm8.py` read,
2. the **list for the last scored Monday** (5 Dec 2011, no outcome yet) and its KPIs.

**Weekly voucher rule (Base scenario: 10% voucher).** Rank the week's active customers by
calibrated churn probability and offer the voucher down the list until the next offer
would exceed the weekly expected discount budget. The budget is the one fixed in
`06_business_impact.ipynb` (Section 7.3), so run notebook 06 first.
""")
code("""
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from policy_lib import expected_margin, weekly_margin

SCEN = dict(d=0.10, e=0.30, m=0.40, r=0.40, c=0.50)  # Base scenario (Section 7.2)
SC = (SCEN["d"], SCEN["e"], SCEN["m"], SCEN["r"], SCEN["c"])
MODEL_VERSION = "lr_extended_c0.01_weighted_cal_h90"
budget_table = pd.read_csv("../data/results/budget_comparison.csv", index_col=0)
BUDGET = budget_table.loc["Highest expected margin per contact", "expected_spend"]  # the Section 7.3 budget
print(f"Weekly expected discount budget: £{BUDGET:,.2f}")

data = pd.read_parquet("../data/processed/modeling_dataset_h90.parquet")
scores = pd.read_parquet("../data/results/churn_scores_h90.parquet")
key = ["customer_id", "snapshot_date"]
assert not scores.duplicated(key).any()
""")
md("## 8.4.1 Join and integrity checks")
code("""
feat = data[key + ["value_v", "rfm_segment"]]
t = scores.rename(columns={"segment": "customer_type"}).merge(
    feat, on=key, how="left", validate="one_to_one", indicator=True)
print(t["_merge"].value_counts().to_dict())
print(t.groupby("split").agg(rows=("customer_id", "size"), weeks=("snapshot_date", "nunique"),
      missing_value=("value_v", lambda s: s.isna().sum()), missing_rfm=("rfm_segment", lambda s: s.isna().sum())))
assert t.value_v.notna().all() and t.rfm_segment.notna().all()
assert t.model_version.nunique() == 1 and t.model_version.iloc[0] == MODEL_VERSION
assert t.churn_probability_calibrated.between(0, 1).all()
t = t.drop(columns="_merge")
""")
md("""
## 8.4.2 Weekly voucher rule

* `expected_discount_cost` = d·V·(p·e + (1 − p)·r), the expected cost of one offer.
* `expected_margin` = E[ΔM] = p·e·V·(m − d) − (1 − p)·r·V·d − c, shown on the list.
* **Voucher**: within the week's ranking by calibrated churn probability (`rank`, 1 = highest
  risk), every customer before the cumulative expected discount cost passes the budget.
* **No voucher**: everyone else.
""")
code("""
d, e, m, r, c = SC
p = t.churn_probability_calibrated
t["expected_discount_cost"] = d * t.value_v * (p * e + (1 - p) * r)
t["expected_margin"] = expected_margin(p, t.value_v, d, e, m, r, c)
t = t.sort_values(["snapshot_date", "rank"]).reset_index(drop=True)
spent = t.groupby("snapshot_date")["expected_discount_cost"].cumsum()
t["action"] = np.where(spent <= BUDGET, "Voucher", "No voucher")
t.groupby(["split", "action"]).agg(rows=("customer_id", "size"), mean_p=("churn_probability_calibrated", "mean"),
                                   mean_value=("value_v", "mean"), churn=("churn", "mean")).round(3)
""")
code("""
# Consistency with Section 7.4: on the test weeks this is the "highest risk of no purchase" list
test = t[t.split == "test"]
mk = test.action == "Voucher"
w = weekly_margin(test, mk, SC)
print(f"Test weeks: {mk.sum() / w.size:,.1f} vouchers per week, simulated net margin £{w.mean():,.2f} per weekly cohort")
assert np.isclose(mk.sum() / w.size, budget_table.loc["Highest risk of no purchase", "contacts_week"])
assert np.isclose(w.mean(), budget_table.loc["Highest risk of no purchase", "net_margin_90d"])
print(f"One-time customers: {(test.customer_type == 'one-time').mean():.1%} of test rows, "
      f"{(test[mk].customer_type == 'one-time').mean():.1%} of the vouchers")
""")
md("""
## 8.4.3 List for the last scored Monday

The last scored Monday has no outcome yet: this is the list marketing would act on.
""")
code("""
last = t[t.snapshot_date == t.snapshot_date.max()]
offer = last[last.action == "Voucher"]
print("Scoring date:", last.snapshot_date.iloc[0].date(), "| active customers:", len(last))
kpi = pd.Series({
    "Active customers": len(last),
    "Customers offered a voucher": len(offer),
    "Expected discount cost of the list (£)": offer.expected_discount_cost.sum(),
    "Expected incremental margin of the list (£, next 90 days)": offer.expected_margin.sum(),
    "Expected share of vouchers sent to likely buyers": 1 - offer.churn_probability_calibrated.mean(),
})
display(kpi.round(3).to_frame("value"))
offer.groupby("rfm_segment").size().sort_values(ascending=False).rename("vouchers").to_frame()
""")
md("## 8.4.4 Export")
code("""
cols = ["customer_id", "snapshot_date", "split", "churn_probability", "churn_probability_calibrated", "risk_band",
        "rank", "value_v", "rfm_segment", "customer_type", "expected_discount_cost", "expected_margin", "action",
        "model_version", "churn"]
out = t[cols].rename(columns={"churn": "churn_actual"})
out["scenario"] = f"Base d=10% e=30% m=40% r=40% c=0.50; weekly budget £{BUDGET:,.0f}"
out.to_parquet("../data/results/customer_scores_actions_h90.parquet", index=False)
print(out.shape, "-> data/results/customer_scores_actions_h90.parquet")
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/07_action_table.ipynb")
