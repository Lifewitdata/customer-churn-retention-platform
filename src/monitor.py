"""Drift monitoring (Phase 10).

Three measurements per batch, because a score-only PSI hides the failure modes
that actually happen in production:

1. score_psi            — drift in the predicted churn_probability distribution.
2. feature_psi          — drift per model input (quantile PSI for numerics, value-share
                          PSI for categoricals), so a shifted feed is attributable.
3. category_diffs       — values present in the batch but never seen in training
                          (OneHotEncoder(handle_unknown="ignore") silently zeroes these).
4. realised discrimination — AUC / PR-AUC / precision of the High tier, computed only
                          when the batch carries observed Churn labels.

Reference distribution is outputs/scored_customers.csv (the snapshot the model was
trained and scored on). Trigger: PSI > 0.25 = retrain / investigate, 0.10-0.25 = watch.

Read outputs/drift_report.csv for what this build actually measured, and
data/raw/ibm_telco_11_1_3/PROVENANCE.md for what the tested batch is and is not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HIGH_RISK_CUTOFF = 0.60


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index on quantile bins of the reference distribution.

    Continuous inputs only - see psi_discrete for categoricals. On a 0/1 indicator the
    reference quantiles collapse to two edges, there is nothing to bin, and this returns
    0.0 regardless of how far the share moved.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(actual, qs)[0] / len(actual)
    e = np.where(e == 0, 1e-4, e)
    a = np.where(a == 0, 1e-4, a)
    return float(np.sum((e - a) * np.log(e / a)))


def psi_discrete(expected, actual, floor: float = 1e-4) -> float:
    """PSI over the share of each observed value, for categorical columns."""
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    cats = sorted(set(e.index) | set(a.index))
    ev = e.reindex(cats, fill_value=0.0).clip(lower=floor)
    av = a.reindex(cats, fill_value=0.0).clip(lower=floor)
    return float(((ev - av) * np.log(ev / av)).sum())


def verdict(value: float) -> str:
    if pd.isna(value):
        return "NOT COMPUTABLE"
    if value > 0.25:
        return "RETRAIN / INVESTIGATE"
    if value > 0.10:
        return "WATCH"
    return "STABLE"


def _frac(series: pd.Series, allowed: set) -> float:
    return float(series.astype(str).isin(allowed).mean())


def category_diffs(reference: pd.DataFrame, batch: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in reference or col not in batch:
            continue
        ref_vals = set(reference[col].dropna().astype(str))
        bat_vals = set(batch[col].dropna().astype(str))
        unseen = sorted(bat_vals - ref_vals)
        missing = sorted(ref_vals - bat_vals)
        share_unseen = float(batch[col].astype(str).isin(unseen).mean()) if unseen else 0.0
        rows.append({"column": col, "n_ref_categories": len(ref_vals), "n_batch_categories": len(bat_vals),
                     "unseen_in_reference": "; ".join(unseen) or "-",
                     "dropped_from_batch": "; ".join(missing) or "-",
                     "share_rows_unseen": round(share_unseen, 4),
                     "coverage_rate": round(_frac(batch[col], ref_vals), 4)})
    return pd.DataFrame(rows)


def feature_drift(reference: pd.DataFrame, batch: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in reference or col not in batch:
            continue
        r, b = pd.to_numeric(reference[col], errors="coerce"), pd.to_numeric(batch[col], errors="coerce")
        if r.notna().sum() == 0 or b.notna().sum() == 0:
            continue
        v = psi(r.dropna(), b.dropna())
        rows.append({"column": col, "kind": "numeric", "psi": round(v, 4), "verdict": verdict(v),
                     "ref_mean": round(float(r.mean()), 4), "batch_mean": round(float(b.mean()), 4),
                     "mean_shift_pct": round(float((b.mean() - r.mean()) / abs(r.mean()) * 100), 2)
                     if r.mean() not in (0, 0.0) else np.nan})
    return pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame(
        columns=["column", "kind", "psi", "verdict", "ref_mean", "batch_mean", "mean_shift_pct"])


def categorical_drift(reference: pd.DataFrame, batch: pd.DataFrame, columns) -> pd.DataFrame:
    """PSI on the value-share distribution of each column, plus the category whose share
    moved most. psi() cannot be used here: it needs enough distinct values to bin."""
    rows = []
    for col in columns:
        if col not in reference or col not in batch:
            continue
        ref_s, bat_s = reference[col].astype(str), batch[col].astype(str)
        v = psi_discrete(ref_s, bat_s)
        e = ref_s.value_counts(normalize=True)
        a = bat_s.value_counts(normalize=True)
        delta = (a - e).abs()
        worst_cat = str(delta.idxmax()) if len(delta) else ""
        rows.append({"column": col, "kind": "categorical", "psi": round(v, 4), "verdict": verdict(v),
                     "worst_category": worst_cat,
                     "ref_share": round(float(e.get(worst_cat, 0.0)), 4),
                     "batch_share": round(float(a.get(worst_cat, 0.0)), 4)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame(
        columns=["column", "kind", "psi", "verdict", "worst_category", "ref_share", "batch_share"])


def realised_performance(batch: pd.DataFrame, label_col: str = "target",
                         prob_col: str = "churn_probability") -> dict:
    """Discrimination measured on the batch, only where outcomes are observed."""
    from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score
    if label_col not in batch or prob_col not in batch:
        return {}
    d = batch[[label_col, prob_col]].dropna()
    y = d[label_col].astype(int).values
    p = d[prob_col].astype(float).values
    if len(np.unique(y)) < 2:
        return {}
    flag = p >= HIGH_RISK_CUTOFF
    return {"batch_rows": int(len(d)), "batch_churn_rate": round(float(y.mean()), 4),
            "realised_roc_auc": round(float(roc_auc_score(y, p)), 4),
            "realised_pr_auc": round(float(average_precision_score(y, p)), 4),
            "realised_f1_at_0.60": round(float(f1_score(y, flag.astype(int), zero_division=0)), 4),
            "high_tier_precision_at_0.60": round(float(precision_score(y, flag.astype(int), zero_division=0)), 4)
            if flag.any() else np.nan,
            "high_tier_share_flagged": round(float(flag.mean()), 4)}


def build_drift_report(reference: pd.DataFrame, batches: dict[str, pd.DataFrame], numeric_cols,
                       cat_cols, notes: dict[str, str] | None = None) -> pd.DataFrame:
    """One tidy row per (batch, measurement) plus the per-column detail tables."""
    notes = notes or {}
    out = []
    for name, batch in batches.items():
        score_v = psi(reference["churn_probability"], batch["churn_probability"])
        row = {"batch": name, "measurement": "score_psi", "value": round(score_v, 4), "verdict": verdict(score_v),
               "detail": notes.get(name, "")}
        out.append(row)
        for df, kind in [(feature_drift(reference, batch, numeric_cols), "feature_psi_numeric"),
                         (categorical_drift(reference, batch, cat_cols), "feature_psi_categorical")]:
            for rec in df.to_dict("records"):
                out.append({"batch": name, "measurement": f"{kind}:{rec['column']}", "value": rec["psi"],
                            "verdict": rec["verdict"], "detail": str({k: v for k, v in rec.items()
                                                                      if k not in ("psi", "verdict", "column",
                                                                                   "kind")})})
        cd = category_diffs(reference, batch, cat_cols)
        for rec in cd.to_dict("records"):
            if rec["unseen_in_reference"] != "-" or rec["dropped_from_batch"] != "-":
                out.append({"batch": name, "measurement": f"category_change:{rec['column']}",
                            "value": rec["share_rows_unseen"], "verdict": "SCHEMA CHANGE",
                            "detail": f"unseen={rec['unseen_in_reference']} dropped={rec['dropped_from_batch']} "
                                      f"coverage={rec['coverage_rate']}"})
        perf = realised_performance(batch)
        for k, v in perf.items():
            out.append({"batch": name, "measurement": k, "value": v, "verdict": "", "detail": "labels observed on batch"})
    return pd.DataFrame(out)


def write_drift_figure(reference: pd.DataFrame, batches: dict[str, pd.DataFrame], fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(reference["churn_probability"], bins=40, density=True, alpha=0.55, label="reference snapshot")
    for i, (name, batch) in enumerate(batches.items()):
        ax.hist(batch["churn_probability"], bins=40, density=True, alpha=0.45,
                label=f"batch: {name}")
    ax.axvline(HIGH_RISK_CUTOFF, ls="--", c="crimson", label=f"High-risk cutoff {HIGH_RISK_CUTOFF}")
    ax.set_xlabel("calibrated churn probability")
    ax.set_ylabel("density")
    ax.set_title("Score distribution: reference vs monitored batches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
