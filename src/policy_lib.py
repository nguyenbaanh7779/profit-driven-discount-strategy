"""Shared helpers for the discount simulation.

Used by notebooks/06_business_impact.ipynb, notebooks/07_action_table.ipynb
and dashboard/app.py. Scenario parameters: d = discount, e = response of
customers who would otherwise stop buying, m = gross margin before discount,
r = redemption by customers who would buy anyway, c = contact cost. V is the
capped customer value `value_v` (report Section 5.5).
"""
import numpy as np
import pandas as pd


def expected_margin(p, v, d, e, m, r, c_contact):
    """E[dM] per customer: p*e*V*(m-d) - (1-p)*r*V*d - c (report Section 7.2)."""
    return p * e * v * (m - d) - (1 - p) * r * v * d - c_contact


def top_k_mask(df, score_col, k, by="snapshot_date"):
    """Boolean mask of the top-k fraction of each week by `score_col`."""
    rank = df.groupby(by)[score_col].rank(ascending=False, method="first")
    size = df.groupby(by)[score_col].transform("size")
    return rank <= np.maximum(np.round(size * k), 1)


def row_gain(df, d, e, m, r, c, vcol="value_v"):
    """Realized margin of contacting each row, using the observed churn label (Section 7.2)."""
    v = df[vcol]
    return np.where(df["churn"] == 1, e * v * (m - d), -r * v * d) - c


def weekly_margin(df, mask, scen, vcol="value_v"):
    """Realized margin of the targeted rows, summed per scoring week."""
    d, e, m, r, c = scen
    g = pd.Series(row_gain(df, d, e, m, r, c, vcol), index=df.index)
    weeks = df["snapshot_date"].unique()
    return (
        g[mask].groupby(df["snapshot_date"][mask]).sum()
        .reindex(weeks, fill_value=0)
    )
