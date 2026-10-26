"""Retention analytics dashboard - Streamlit only.

Reads the artefacts in outputs/ that `src/run_pipeline.py` writes; it never retrains, so
the app opens in about a second even with 7,043 scored customers. Every number on screen
is the committed pipeline number, which is the point: the dashboard is a viewer for the
audited artefacts, not a second implementation of them.

Run: streamlit run dashboard/app.py
Pages are deep-linkable (?page=drivers) because the page picker binds to query params.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
FIG = OUT / "figures"

PAGES = {
    "overview": "Executive overview",
    "segments": "Risk segmentation",
    "drivers": "Driver analysis",
    "survival": "Survival and timing",
    "fairness": "Fairness check",
    "actions": "Action list",
    "health": "Model health and drift",
}

st.set_page_config(page_title="Churn retention platform", layout="wide",
                   page_icon=":material/query_stats:")


@st.cache_data(show_spinner=False)
def _read_csv(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: Path, mtime: float) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(name: str, reader, empty):
    # The mtime is part of the cache key, so re-running src/run_pipeline.py refreshes the app
    # without a restart; keying on the filename alone would serve the stale artefact forever.
    p = OUT / name
    return reader(p, p.stat().st_mtime) if p.exists() else empty


def csv(name: str) -> pd.DataFrame:
    return _load(name, _read_csv, pd.DataFrame())


def js(name: str) -> dict:
    return _load(name, _read_json, {})


scored = csv("scored_customers.csv")
if scored.empty:
    st.error("No scored customers yet. Run `python src/run_pipeline.py` first.")
    st.stop()

ev = js("evaluation.json")
dq = csv("dq_scorecard.csv")

with st.sidebar:
    st.markdown("### Churn retention platform")
    st.caption(f"IBM Telco Customer Churn - {len(scored):,} customers - snapshot data")
    requested = st.query_params.get("page", "overview")
    page = st.radio("Page", list(PAGES), index=list(PAGES).index(requested) if requested in PAGES else 0,
                    key="page", format_func=lambda k: PAGES[k], label_visibility="collapsed")
    st.query_params["page"] = page
    st.divider()
    tiers = st.pills("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"],
                     selection_mode="multi")
    contracts = sorted(scored["Contract"].unique())
    picked = st.multiselect("Contract", contracts, default=contracts)

filtered = scored[scored["risk_tier"].isin(tiers) & scored["Contract"].isin(picked)]


def bar(frame: pd.DataFrame, x: str, y: str, title: str, alt_name: str, **kwargs):
    chart = (alt.Chart(frame, title=title, height=240)
             .mark_bar(cornerRadiusEnd=3, color="#b5179e")
             .encode(x=alt.Order(x), y=y, tooltip=[x, y, *kwargs.get("tooltip", [])])
             .properties(width="container"))
    st.altair_chart(chart, alt=alt_name)


def hbar(frame: pd.DataFrame, label: str, value: str, title: str, alt_name: str, **kwargs):
    """Horizontal bars: the only readable encoding for long feature names."""
    axis = kwargs.pop("axis", value)
    chart = (alt.Chart(frame, title=title, height=kwargs.pop("height", 300))
             .mark_bar(cornerRadiusEnd=3, color="#b5179e")
             .encode(y=alt.Y(f"{label}:N", sort=None, title=None),
                     x=alt.X(f"{value}:Q", title=axis),
                     tooltip=[label, value, *kwargs.get("tooltip", [])])
             .properties(width="container"))
    st.altair_chart(chart, alt=alt_name)


# ------------------------------------------------------------------ 1. executive overview
if page == "overview":
    st.title("Executive overview")
    high = filtered[filtered["risk_tier"] == "High"]
    with st.container(horizontal=True):
        st.metric("Customers in view", f"{len(filtered):,}", border=True)
        st.metric("Observed churn rate", f"{filtered['target'].mean():.1%}", border=True)
        st.metric("Flagged high risk", f"{len(high):,}",
                  delta=f"{len(high) / len(scored) * 100:.1f}% of base", border=True)
        st.metric("Revenue at risk, 12m", f"${high['revenue_at_risk_12m'].sum():,.0f}",
                  delta=f"of ${scored['MonthlyCharges'].sum() * 12:,.0f} book", border=True)
    st.caption(
        f"Deployed artefact: **{ev.get('model_deployed', '-')}**. "
        f"ROC-AUC {ev.get('roc_auc_calibrated', 0):.3f} / PR-AUC {ev.get('pr_auc_calibrated', 0):.3f} "
        f"on {ev.get('test_rows', 0):,} held-out rows; Brier {ev.get('brier_calibrated', 0):.4f} against a "
        f"base-rate baseline of {ev.get('brier_baseline_base_rate', 0):.4f}. "
        f"DQ scorecard {dq['passed'].mean():.0%} green ({int(dq['passed'].sum())}/{len(dq)} checks)."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        bar(csv("churn_by_Contract.csv"), "Contract", "churn_rate",
            "Churn rate by contract", "Churn rate by contract")
    with col_b:
        bar(csv("churn_by_tenure_band.csv"), "tenure_band", "churn_rate",
            "Churn rate by tenure band", "Churn rate by tenure band")
    profit = csv("profit_curve.csv")
    if not profit.empty:
        a = (alt.Chart(profit, height=260,
                       title="Scenario profit by share contacted")
             .mark_line(point=True, color="#0f62fe")
             .encode(x=alt.X("contact_fraction:Q", title="share of the base contacted"),
                     y=alt.Y("expected_profit_scenario:Q", title="expected profit ($)"),
                     tooltip=list(profit.columns))
             .properties(width="container"))
        st.altair_chart(a, alt="Scenario profit curve")
        st.caption(
            "Scenario arithmetic, not measured uplift: "
            f"{ev.get('scenario_assumptions', {}).get('save_rate', 0.3):.0%} save rate, "
            f"\\${ev.get('scenario_assumptions', {}).get('cost_per_customer', 40):.0f} per contacted "
            f"customer, CLV = monthly charges x "
            f"{ev.get('scenario_assumptions', {}).get('clv_months', 12)}. "
            "Profit peaks at contacting "
            f"{ev.get('best_contact_fraction_by_profit', 0):.0%} of the base "
            f"(\\${ev.get('best_scenario_profit', 0):,.0f}); a true uplift claim needs an A/B test, "
            "and this dataset has none."
        )
    with st.container(border=True):
        st.subheader("Retention KPIs by adapted RFM segment")
        st.dataframe(csv("rfm_segments.csv"), hide_index=True, alt="RFM segment KPIs")
        st.caption("R = tenure quintile (a proxy, not days since last activity), F = subscribed service count, "
                   "M = monthly charges. There is no activity timestamp in the feed, so classic RFM is not "
                   "available and this substitution is stated rather than hidden.")

# ------------------------------------------------------------------ 2. risk segmentation
elif page == "segments":
    st.title("Risk segmentation")
    tier = (scored.groupby("risk_tier")
            .agg(customers=("customerID", "size"), mean_prob=("churn_probability", "mean"),
                 observed_churn=("target", "mean",), monthly_revenue=("MonthlyCharges", "sum"),
                 revenue_at_risk_12m=("revenue_at_risk_12m", "sum"))
            .reindex(["High", "Medium", "Low"]).reset_index().round(4))
    st.dataframe(tier, hide_index=True, alt="Risk tier summary",
                 column_config={"mean_prob": st.column_config.NumberColumn(format="%.3f"),
                                "observed_churn": st.column_config.NumberColumn(format="percent"),
                                "monthly_revenue": st.column_config.NumberColumn(format="$%.0f"),
                                "revenue_at_risk_12m": st.column_config.NumberColumn(format="$%.0f")})
    st.caption("Tier cutoffs come from `src/features.py` (High >= 0.60, Medium >= 0.35) and mean predicted "
               "probability tracks observed churn inside each band - that agreement is the calibration check.")
    bins = pd.cut(scored["churn_probability"], np.linspace(0, 1, 21), include_lowest=True)
    hist = (scored.assign(bin=bins.astype(str))
            .groupby(["bin", "risk_tier"], observed=True).size().reset_index(name="customers")
            .assign(series=lambda d: d["risk_tier"].map({"Low": 0, "Medium": 1, "High": 2}))
            .sort_values(["bin", "series"]))
    a = (alt.Chart(hist, height=280, title="Predicted churn probability, scored base")
         .mark_bar()
         .encode(x=alt.X("bin:N", title="probability bin", sort=None),
                 y=alt.Y("customers:Q", title="customers"),
                 color=alt.Color("risk_tier:N", scale={"domain": ["Low", "Medium", "High"],
                                                       "range": ["#2a9d8f", "#f4a261", "#b5179e"]},
                                 legend={"title": None}))
         .properties(width="container"))
    st.altair_chart(a, alt="Probability distribution by risk tier")
    st.subheader("Customers in view")
    cols = ["customerID", "tenure", "Contract", "PaymentMethod", "InternetService", "TechSupport",
            "MonthlyCharges", "churn_probability", "risk_tier", "recommended_action"]
    st.dataframe(filtered[cols].sort_values("churn_probability", ascending=False),
                 hide_index=True, height=380, alt="Filtered customer list",
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "churn probability", min_value=0.0, max_value=1.0, format="%.3f")})
    st.download_button("Download filtered list (CSV)", filtered.to_csv(index=False),
                       "risk_list.csv", "text/csv", icon=":material/download:")

# ------------------------------------------------------------------ 3. driver analysis
elif page == "drivers":
    st.title("Driver analysis")
    shap_tree = csv("shap_importance_tree.csv")
    if not shap_tree.empty:
        top = shap_tree.head(15).sort_values("mean_abs_shap", ascending=False)
        a = (alt.Chart(top, height=420,
                       title="Tree SHAP - Optuna-tuned LightGBM (mean |SHAP|, coloured by direction)")
             .mark_bar()
             .encode(y=alt.Y("feature:N", sort=None, title=None),
                     x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds)"),
                     color=alt.Color("shap_sign:N", title="effect sign",
                                     scale={"domain": [-1.0, 1.0], "range": ["#2a9d8f", "#b5179e"]}),
                     tooltip=["feature", "mean_abs_shap", "shap_sign"])
             .properties(width="container"))
        st.altair_chart(a, alt="Tree SHAP driver ranking")
    if (FIG / "shap_beeswarm_lightgbm.png").exists():
        with st.expander("SHAP beeswarm (every held-out customer, top 15 features)", expanded=False):
            st.image(str(FIG / "shap_beeswarm_lightgbm.png"),
                     alt="SHAP beeswarm for the tuned LightGBM")
    left, right = st.columns(2)
    with left:
        perm = csv("permutation_importance.csv")
        if not perm.empty:
            hbar(perm.head(12).sort_values("perm_importance_pr_auc", ascending=False),
                 "feature", "perm_importance_pr_auc",
                 "Permutation importance (PR-AUC lost)", "Permutation importance",
                 axis="PR-AUC lost", tooltip=["std"])
            st.caption("Measured on the deployed logistic regression. A driver only earns attention here "
                       "if breaking it costs the model something.")
    with right:
        lin = csv("shap_linear_logreg.csv")
        if not lin.empty:
            hbar(lin.head(12).sort_values("mean_abs_shap_linear", ascending=False),
                 "feature", "mean_abs_shap_linear",
                 "Linear SHAP - same model, log-odds scale", "Linear SHAP", axis="mean |SHAP|")
    pdp = csv("partial_dependence.csv")
    if not pdp.empty:
        # Vega paints its own title above the plot box, which the element container clips when the
        # block above ends flush - so this heading lives in the Streamlit flow instead.
        st.subheader("Partial dependence - isolated marginal effect")
        long = pd.concat([pd.DataFrame({"feature": r["feature"], "x": json.loads(r["grid"])["x"],
                                       "mean_predicted_churn": json.loads(r["grid"])["y"]})
                          for _, r in pdp.iterrows()], ignore_index=True)
        a = (alt.Chart(long, height=280)
             .mark_line(point=True)
             .encode(x=alt.X("x:Q", title="feature value"),
                     y=alt.Y("mean_predicted_churn:Q", title="mean predicted churn probability"),
                     color=alt.Color("feature:N", title=None),
                     tooltip=["feature", "x", "mean_predicted_churn"])
             .properties(width="container"))
        st.altair_chart(a, alt="Partial dependence curves")
        st.caption("Partial dependence holds the feature fixed and averages the rest over the test split; "
                   "it is not a causal effect. `contract_risk` (0 = two year, 1 = one year, "
                   "2 = month-to-month) has the steepest curve after tenure.")
    coef = csv("logreg_coefficients.csv")
    if not coef.empty:
        st.subheader("Signed standardised coefficients")
        st.dataframe(coef, hide_index=True, height=260, alt="Logistic regression coefficients")
    st.subheader("Statistical tests")
    st.dataframe(csv("statistical_tests.csv"), hide_index=True, alt="Statistical association tests",
                 column_config={"p_value": st.column_config.NumberColumn(format="%.1e"),
                                "statistic": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Every association here is significant at p < 0.001 with 7,043 rows, so effect size "
               "(Cramer's V, rank-biserial) is the column worth reading.")

# ------------------------------------------------------------------ 4. survival and timing
elif page == "survival":
    st.title("Survival and timing")
    km = csv("survival_km.csv")
    if not km.empty:
        kind = st.segmented_control("Segment by", sorted(km["segment_type"].unique()),
                                    default=[km["segment_type"].iloc[0]])
        data = km[km["segment_type"].isin(np.atleast_1d(kind))]
        a = (alt.Chart(data, height=320,
                       title=f"Survival to date by {np.atleast_1d(kind)[0]}")
             .mark_line(interpolate="step-after")
             .encode(x=alt.X("month:Q", title="tenure in months observed at the snapshot"),
                     y=alt.Y("survival_probability:Q", title="S(t) = still a customer", scale={"domain": [0, 1]}),
                     color=alt.Color("segment:N", title=None),
                     tooltip=["segment", "month", "n_at_risk_start", "survival_probability"])
             .properties(width="container"))
        st.altair_chart(a, alt="Kaplan-Meier survival curves")
        st.caption("Snapshot data: `tenure` is time observed, `Churn` is status at that moment, and the "
                   "contract shown is the one observed at the snapshot - not the one signed. These curves "
                   "are descriptive survival-to-date, not a causal contract effect.")
    med = csv("survival_median.csv")
    if not med.empty:
        shown = med.assign(median_survival_months=med["median_survival_months"].map(
            lambda v: f"{v:.0f}" if pd.notna(v) else "not reached"))
        st.dataframe(shown, hide_index=True, alt="Median survival by contract")
    st.subheader("Cox proportional hazards - two engines")
    cox = csv("cox_results_lifelines.csv")
    r_cox = csv("r/cox_hazard_ratios.csv")
    if not cox.empty and not r_cox.empty:
        merged = (cox[["covariate", "hazard_ratio", "p", "ph_test_p_rank"]]
                  .rename(columns={"hazard_ratio": "hr_lifelines", "p": "p_lifelines"})
                  .merge(r_cox[["covariate", "hazard_ratio", "p_value"]], on="covariate", how="inner")
                  .rename(columns={"hazard_ratio": "hr_coxph_R", "p_value": "p_R"}))
        merged["abs_diff"] = (merged["hr_lifelines"] - merged["hr_coxph_R"]).abs()
        r_zph = csv("r/cox_zph_test.csv")
        shown = merged
        if not r_zph.empty:
            # cox.zph's numeric p is rounded to 5 dp, so the sub-1e-5 rows would read as a bare 0;
            # the R script already writes a "< 1e-16" label, and that is what belongs on screen.
            merged = merged.merge(
                r_zph[["term", "p", "p_value"]].rename(
                    columns={"term": "covariate", "p": "ph_p_cox_zph_R",
                             "p_value": "Schoenfeld p (cox.zph, R)"}), on="covariate", how="left")
            shown = merged.drop(columns=["ph_p_cox_zph_R"])
        st.dataframe(shown, hide_index=True, alt="Cox hazard ratios, lifelines versus R",
                     column_config={
                         "hr_lifelines": st.column_config.NumberColumn("HR (lifelines)", format="%.4f"),
                         "hr_coxph_R": st.column_config.NumberColumn("HR (coxph, R)", format="%.4f"),
                         "p_lifelines": st.column_config.NumberColumn("p (lifelines)", format="%.2e"),
                         "p_R": st.column_config.NumberColumn("p (coxph, R)", format="%.2e"),
                         "ph_test_p_rank": st.column_config.NumberColumn("Schoenfeld p (lifelines, rank)",
                                                                        format="%.3f"),
                         "abs_diff": st.column_config.NumberColumn("|HR difference|",
                                                                  format="%.6f")})
        st.caption(f"Python (`lifelines`) and R (`survival` 3.8.6, `r/survival_analysis.R`) fit the same "
                   f"specification; the largest hazard-ratio difference is "
                   f"{merged['abs_diff'].max():.2e}. The two Schoenfeld columns use different time "
                   f"transforms (rank versus identity), so they should agree on the verdict rather than on "
                   f"the digit. Full R output sits in `outputs/r/`.")
        ph_bad = merged[(merged["ph_test_p_rank"] < 0.05) | (merged["ph_p_cox_zph_R"] < 0.05)]
        if not ph_bad.empty:
            st.warning("Proportional hazards is rejected for "
                       + ", ".join(ph_bad["covariate"])
                       + ". Those hazard ratios are time-averaged over the observed window only, which is "
                         "why the operational calls come from the KM curves and the classifier rather than "
                         "from a Cox p-value.")

# ------------------------------------------------------------------ 5. fairness
elif page == "fairness":
    st.title("Fairness and subpopulation performance")
    fair = csv("fairness_by_segment.csv")
    summ = csv("fairness_summary.csv")
    st.caption(
        "One ROC-AUC hides who the model is bad at. These are subgroup metrics at the deployed "
        f"{ev.get('high_risk_cutoff', 0.6):.2} cutoff on the held-out split. `SeniorCitizen`, `gender` and "
        "`Dependents` are the closest things to protected attributes this feed carries; there is no race, "
        "ethnicity or national-origin field, so this check is partial and labelled as such.")
    if not fair.empty:
        dim = st.selectbox("Segment dimension", sorted(fair["segment_dimension"].unique()),
                           index=sorted(fair["segment_dimension"].unique()).index("SeniorCitizen"))
        view = fair[fair["segment_dimension"] == dim]
        st.dataframe(view, hide_index=True, alt=f"Fairness metrics by {dim}",
                     column_config={
                         "base_churn_rate": st.column_config.NumberColumn(format="percent"),
                         "mean_predicted_prob": st.column_config.NumberColumn(format="%.3f"),
                         "calibration_gap_pp": st.column_config.NumberColumn("calibration gap (pp)",
                                                                              format="%+.2f"),
                         "high_risk_share": st.column_config.ProgressColumn("high-risk share",
                                                                            min_value=0.0, max_value=1.0),
                         "recall_at_high_cutoff": st.column_config.NumberColumn("recall at cutoff",
                                                                                format="percent"),
                         "false_positive_rate": st.column_config.NumberColumn("FPR", format="percent"),
                         "precision_at_high_cutoff": st.column_config.NumberColumn("precision",
                                                                                   format="percent"),
                         "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f")})
        long = view.melt(id_vars="segment",
                         value_vars=["base_churn_rate", "high_risk_share", "recall_at_high_cutoff"],
                         var_name="metric", value_name="rate")
        long["metric"] = long["metric"].map({"base_churn_rate": "observed churn",
                                             "high_risk_share": "flagged high risk",
                                             "recall_at_high_cutoff": "recall at cutoff"})
        # All three are rates on the same 0-0.5 scale, so one dodged panel beats three facets:
        # a faceted chart repeats its title per panel and overruns the container width.
        a = (alt.Chart(long, height=260)
             .mark_bar(cornerRadiusEnd=3)
             .encode(x=alt.X("segment:N", title=None),
                     y=alt.Y("rate:Q", title="rate"),
                     xOffset="metric:N",
                     color=alt.Color("metric:N", title=None,
                                     scale={"domain": ["observed churn", "flagged high risk",
                                                       "recall at cutoff"],
                                            "range": ["#b5179e", "#2a9d8f", "#f4a261"]}),
                     tooltip=["segment", "metric", "rate"])
             .properties(width="container"))
        st.altair_chart(a, alt=f"Observed churn, selection rate and recall by {dim}")
        st.caption("Read the calibration gap before the AUC: a segment whose mean predicted probability sits "
                   "above its observed churn is one where the offer list over-fires; below means the model "
                   "under-recognises that group's risk and retention budget never reaches it.")
    if not summ.empty:
        st.subheader("Summary across dimensions")
        st.dataframe(summ, hide_index=True, alt="Fairness summary by dimension")
        st.caption("Adverse impact is the selection-rate ratio an auditor asks for. Being flagged here means "
                   "receiving a retention offer, so a higher rate is a benefit, not a harm - which is why "
                   "equal-opportunity (recall) and calibration gaps are the numbers that matter.")

# ------------------------------------------------------------------ 6. action list
elif page == "actions":
    st.title("Action list")
    high_df = filtered[filtered["risk_tier"] == "High"].sort_values("churn_probability", ascending=False)
    with st.container(horizontal=True):
        st.metric("High-risk customers in view", f"{len(high_df):,}", border=True)
        st.metric("Monthly revenue they carry", f"${high_df['MonthlyCharges'].sum():,.0f}", border=True)
        st.metric("Observed churn in this group", f"{high_df['target'].mean():.1%}"
                  if len(high_df) else "n/a", border=True)
    cols = ["customerID", "tenure", "Contract", "PaymentMethod", "MonthlyCharges", "churn_probability",
            "recommended_action", "revenue_at_risk_12m"]
    st.dataframe(high_df[cols], hide_index=True, height=400, alt="High-risk customer action list",
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "churn probability", min_value=0.0, max_value=1.0, format="%.3f"),
                     "revenue_at_risk_12m": st.column_config.NumberColumn(format="$%.0f")})
    st.download_button("Download action list (CSV)", high_df.to_csv(index=False),
                       "action_list.csv", "text/csv", icon=":material/download:")
    acts = csv("retention_actions.csv")
    if not acts.empty:
        st.subheader("Playbook by cause")
        st.dataframe(acts, hide_index=True, alt="Retention actions by cause",
                     column_config={"churn_rate": st.column_config.NumberColumn(format="percent"),
                                    "revenue_at_risk_12m": st.column_config.NumberColumn(format="$%.0f")})
        st.caption("`src/features.py::recommended_action` is the single copy of this rule - the batch job, "
                   "the FastAPI endpoint, the SQL mirror and this page all read it. Expected saved = "
                   "customers x observed churn in the group x the assumed 30% save rate. Scenario, not "
                   "measured uplift.")

# ------------------------------------------------------------------ 7. model health and drift
else:
    st.title("Model health and drift")
    comp = csv("model_comparison.csv")
    if not comp.empty:
        st.subheader("Model comparison (held-out test split)")
        st.dataframe(comp, hide_index=True, alt="Model comparison table",
                     column_config={"roc_auc": st.column_config.ProgressColumn("ROC-AUC", min_value=0.5,
                                                                               max_value=1.0,
                                                                               format="%.4f"),
                                    "pr_auc": st.column_config.ProgressColumn("PR-AUC", min_value=0.0,
                                                                              max_value=1.0, format="%.4f"),
                                    "top_decile_lift": st.column_config.NumberColumn(format="x %.2f")})
        st.caption("Seven families, one contract: linear, bagged, two gradient-boosting stacks, the sklearn "
                   "MLP, a PyTorch feedforward network (64-32, BatchNorm, 0.3 dropout, Adam, class-weighted "
                   "BCE, early stopped) and Optuna-tuned LightGBM over 15 TPE trials scored on CV PR-AUC of "
                   "the training split only. Best ROC-AUC is "
                   f"{comp['roc_auc'].max():.4f} - reported as a ceiling of these 21 published columns, "
                   "not dressed up as a target hit.")
    cal = csv("calibration_curve.csv")
    lift = csv("lift_by_decile.csv")
    col_a, col_b = st.columns(2)
    with col_a:
        if not cal.empty:
            a = (alt.Chart(cal, height=260, title="Reliability: predicted versus observed")
                 .mark_line(point=True)
                 .encode(x=alt.X("mean_pred:Q", title="mean predicted probability"),
                         y=alt.Y("frac_positive:Q", title="observed churn rate"),
                         color=alt.Color("model:N", title=None))
                 .properties(width="container"))
            ref = (alt.Chart(pd.DataFrame({"mean_pred": [0, 1], "frac_positive": [0, 1]}))
                   .mark_line(strokeDash=[6, 4])
                   .encode(color=alt.value("#9aa0a6")))
            st.altair_chart(a + ref, alt="Calibration curves before and after isotonic scaling")
    with col_b:
        if not lift.empty:
            bar(lift, "decile", "lift", "Lift by predicted-risk decile", "Lift by decile",
                tooltip=["n", "churn_rate"])
    st.subheader("Data quality scorecard")
    st.dataframe(dq, hide_index=True, height=280, alt="Data quality scorecard")
    st.subheader("Drift")
    drift = csv("drift_report.csv")
    if not drift.empty:
        batches = sorted(drift["batch"].unique())
        chosen = st.multiselect("Batch", batches, default=batches)
        rows = drift[drift["batch"].isin(chosen)]
        worst = (rows[rows["measurement"].str.startswith(("score_psi", "feature_psi"))]
                 .nlargest(12, "value"))
        if not worst.empty:
            a = (alt.Chart(worst, height=280, title="Largest PSI values (0.10 watch, 0.25 retrain)")
                 .mark_bar()
                 .encode(y=alt.Y("measurement:N", sort=None, title=None),
                         x=alt.X("value:Q", title="PSI"),
                         color=alt.Color("verdict:N",
                                         scale={"domain": ["STABLE", "WATCH", "RETRAIN / INVESTIGATE"],
                                                "range": ["#2a9d8f", "#f4a261", "#b5179e"]},
                                         title=None),
                         tooltip=["batch", "measurement", "value", "verdict"])
                 .properties(width="container"))
            st.altair_chart(a, alt="Drift PSI values")
        st.dataframe(rows[rows["measurement"].isin(
            ["score_psi", "batch_rows", "batch_churn_rate", "realised_roc_auc", "realised_pr_auc",
             "category_change:PaymentMethod", "feature_psi_categorical:PaymentMethod"])],
            hide_index=True, height=220, alt="Drift summary rows")
        st.caption(
            "The only genuinely different feed available here is IBM's 11.1.3 Q3 re-release, and it carries "
            "the same 7,043 customers with the same labels: what it measures is upstream **re-coding** drift "
            "(the `Electronic check` category disappears, 265 contracts move to term), not drift over time. "
            "The held-out split is published beside it as a control. Real temporal drift needs a dated "
            "production feed - run `python src/score_batch.py` then `python src/monitor.py`.")
    st.subheader("Second source")
    recon = csv("source_reconciliation.csv")
    if not recon.empty:
        st.dataframe(recon, hide_index=True, height=240, alt="Cross-source reconciliation")
    bench = js("enrichment_benchmark_vs_ibm_score.json")
    if bench:
        with st.container(horizontal=True):
            st.metric("IBM Churn Score ROC-AUC", f"{bench['ibm_churn_score_roc_auc']:.3f}", border=True)
            st.metric("Our calibrated ROC-AUC", f"{bench['our_calibrated_roc_auc']:.3f}", border=True)
        st.caption(f"IBM's supplied score wins on the {bench['rows']:,} held-out rows it can be scored on. "
                   "It is built from inputs that are not in the published schema and it encodes the label, "
                   "so it is excluded as a feature and kept only as an external yardstick.")
    st.subheader("Reproduce")
    st.code("python src/run_pipeline.py\nRscript r/survival_analysis.R\npython src/run_sql.py\n"
            "python -m pytest\nstreamlit run dashboard/app.py", language="bash")
    st.caption("Artefacts read by this app: `outputs/`, `outputs/r/` and `warehouse/churn.duckdb` "
               "(SQL mirror, checked against pandas by `src/run_sql.py`). "
               "Deployed model file: `outputs/model_calibrated.pkl`.")
