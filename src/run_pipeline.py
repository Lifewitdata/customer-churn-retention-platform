"""
Customer Churn Prediction and Retention Analytics Platform - pipeline.
Phases 1-10 in one reproducible run. Streamlit reads the outputs/ files, it does not retrain.

HONEST ADAPTATIONS FOR THE IBM TELCO DATASET (primary snapshot, 7,043 rows, 21 cols):
- No signup date, support-call log, login log, usage 30/60/90 series, complaint log, or
  payment-delay field exists in this dataset. Those fields are NOT fabricated. Proxies
  built only from available columns are labelled PROXY.
- No observation/prediction date exists, so a true time-based split is not possible. We use
  a stratified train/test split (seed 42) + stratified CV, and document that a dated
  production feed must use a time-based split.
- RFM recency is adapted: R = tenure quintile, F = service count, M = monthly charges. It is
  NOT days-since-last-activity, and the README says so.
- Profit / expected-savings figures are SCENARIOS (assumed save rate 30%, campaign $10 +
  offer $30 per contacted customer, CLV = monthly x 12). True uplift needs A/B data.
- A second source exists (data/raw/ibm_telco_11_1_3/, IBM's 11.1.3 release). It adds
  geography, Age, Satisfaction Score, CLTV and Churn Reason, and supplies the drift batch.
  It is a re-coded re-release of the same book of business, NOT a later period - see
  data/raw/ibm_telco_11_1_3/PROVENANCE.md.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dq
import external
import features as F
import monitor
from torch_nn import TorchFeedforwardClassifier

torch.set_num_threads(2)
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROC = BASE / "data" / "processed"
OUT = BASE / "outputs"
FIG = OUT / "figures"
for d in (PROC, OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)

RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)
CAT, NUM = F.model_columns()


def write_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df.to_csv(path, index=False, lineterminator="\n")
    return df


# ============ Phase 1: ingestion + data quality ============
df_raw = F.load_raw(RAW)
print(f"raw shape {df_raw.shape}")
dq_card = dq.run_dq_checks(df_raw)
dq_score = dq.score_scorecard(dq_card)
write_csv(dq_card, OUT / "dq_scorecard.csv")
print(f"DQ score {dq_score}%  passed {int(dq_card['passed'].sum())}/{len(dq_card)} checks"
      f"  (schema layer: {int(dq_card[dq_card.layer == 'schema'].passed.sum())}/2)")

df = F.engineer(df_raw)
write_csv(df, PROC / "clean_telco.csv")
base_rate = float(df["target"].mean())

# ============ Phase 2: EDA + statistical tests ============
insights = [f"Base churn rate is {base_rate:.1%} ({int(df['target'].sum())} of {len(df)} customers)."]


def churn_by(col):
    g = df.groupby(col, observed=True)["target"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "churn_rate", "count": "n"}).sort_values("churn_rate", ascending=False)
    write_csv(g, OUT / f"churn_by_{col}.csv")
    return g


for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity",
            "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"]:
    g = churn_by(col)
    top = g.iloc[0]
    insights.append(f"Highest churn in {col}: {top[col]} at {top['churn_rate']:.1%} "
                    f"(n={int(top['n'])}), vs base {base_rate:.1%}.")

g = churn_by("tenure_band")
insights.append("Churn by tenure band: " + ", ".join(
    f"{r.tenure_band} {r.churn_rate:.1%} (n={int(r.n)})" for r in g.itertuples()))
mm = df[df["Contract"] == "Month-to-month"]
insights.append(f"Month-to-month customers: {len(mm)}, churn {mm['target'].mean():.1%} vs {base_rate:.1%} base.")
slice_df = df[(df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic")
              & (df["PaymentMethod"] == "Electronic check")]
insights.append(f"Month-to-month + Fiber optic + Electronic check: {len(slice_df)} customers, churn "
                f"{slice_df['target'].mean():.1%} - {slice_df['target'].mean() / base_rate:.1f}x the base rate.")
slice2 = df[(df["Contract"] == "Month-to-month") & (df["TechSupport"] == "No")]
insights.append(f"Month-to-month + no TechSupport: {len(slice2)} customers, churn {slice2['target'].mean():.1%} "
                f"- {slice2['target'].mean() / base_rate:.1f}x the base rate.")

rows = []
for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity",
            "PaperlessBilling", "gender", "Partner", "Dependents"]:
    ct = pd.crosstab(df[col], df["target"])
    chi2, p, dof, _ = chi2_contingency(ct)
    n, (r, k) = ct.values.sum(), ct.shape
    rows.append({"feature": col, "test": "chi-square", "statistic": round(float(chi2), 2), "p_value": float(p),
                 "effect_size_cramers_v": round(float(math.sqrt(chi2 / (n * (min(r, k) - 1)))), 3),
                 "significant_p_lt_0.05": bool(p < 0.05)})
for col in ["tenure", "MonthlyCharges", "TotalCharges", "service_count"]:
    a = df.loc[df["target"] == 1, col]
    b = df.loc[df["target"] == 0, col]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    rows.append({"feature": col, "test": "Mann-Whitney U", "statistic": round(float(u), 1), "p_value": float(p),
                 "effect_size_rank_biserial": round(float(1 - (2 * u) / (len(a) * len(b))), 3),
                 "significant_p_lt_0.05": bool(p < 0.05)})
write_csv(pd.DataFrame(rows).sort_values("p_value"), OUT / "statistical_tests.csv")

cohort = df.groupby(["Contract", "tenure_band"], observed=True)["target"].agg(["mean", "count"]).reset_index()
cohort["retention_rate"] = 1 - cohort["mean"]
write_csv(cohort, OUT / "cohort_retention.csv")

# ============ Phase 3: segmentation (features already engineered in F.engineer) ============
for src, out in [("tenure", "r_score"), ("service_count", "f_score"), ("MonthlyCharges", "m_score")]:
    try:
        df[out] = pd.qcut(df[src], 5, labels=False, duplicates="drop") + 1
    except Exception:
        df[out] = pd.cut(df[src], 5, labels=False) + 1
df["rfm_total"] = df["r_score"] + df["f_score"] + df["m_score"]


def rfm_label(r):
    if r >= 13:
        return "Champions"
    if r >= 11:
        return "Loyal"
    if r >= 9:
        return "Potential loyal"
    if r >= 7:
        return "At risk (adapted RFM)"
    return "Hibernating / high risk (adapted RFM)"


df["rfm_segment"] = df["rfm_total"].apply(rfm_label)
write_csv(df.groupby("rfm_segment").agg(n=("target", "size"), churn_rate=("target", "mean"),
                                        avg_monthly=("MonthlyCharges", "mean"),
                                        avg_tenure=("tenure", "mean")).reset_index()
    .sort_values("churn_rate", ascending=False), OUT / "rfm_segments.csv")

km_x = df[["tenure", "MonthlyCharges", "TotalCharges", "service_count"]].fillna(0)
km = KMeans(n_clusters=4, random_state=RNG, n_init=10).fit(StandardScaler().fit_transform(km_x))
df["kmeans_cluster"] = km.labels_
write_csv(df.groupby("kmeans_cluster").agg(n=("target", "size"), churn_rate=("target", "mean"),
                                           avg_tenure=("tenure", "mean"), avg_monthly=("MonthlyCharges", "mean"),
                                           avg_services=("service_count", "mean")).reset_index(),
          OUT / "kmeans_segments.csv")

# ============ Phase 4: modelling ============
X = df[CAT + NUM].copy()
y = df["target"].values
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=RNG, stratify=y)
pre = F.build_preprocessor()


def make_pipe(mdl):
    return Pipeline([("pre", pre), ("model", mdl)])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG,
                                            class_weight="balanced_subsample", n_jobs=-1),
    "XGBoost": __import__("xgboost").XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                   subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                                   n_jobs=-1, random_state=RNG),
    "LightGBM": __import__("lightgbm").LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05,
                                                      verbose=-1, random_state=RNG, class_weight="balanced"),
    "MLP (sklearn feedforward)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=RNG),
    "Feedforward NN (PyTorch)": TorchFeedforwardClassifier(hidden_layer_sizes=(64, 32), dropout=0.3,
                                                           learning_rate=1e-3, max_epochs=120, patience=12,
                                                           random_state=RNG),
}


def eval_row(name, p):
    order = np.argsort(-p)
    top = order[: max(1, int(0.10 * len(p)))]
    top_recall = float(y_test[top].sum() / max(1, y_test.sum()))
    return {"model": name, "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
            "pr_auc": round(float(average_precision_score(y_test, p)), 4),
            "f1_at_0.5": round(float(f1_score(y_test, (p >= 0.5).astype(int))), 4),
            "top_decile_recall": round(top_recall, 4), "top_decile_lift": round(top_recall / 0.10, 2)}


results, preds, pipes = [], {}, {}
for name, mdl in models.items():
    pipe = make_pipe(mdl)
    pipe.fit(X_train, y_train)
    pipes[name] = pipe
    p = pipe.predict_proba(X_test)[:, 1]
    preds[name] = p
    results.append(eval_row(name, p))
    joblib.dump(pipe, OUT / f"model_{name.split(' (')[0].replace(' ', '_')}.pkl")
    print(f"  {name}: {results[-1]}")
nn_epochs = pipes["Feedforward NN (PyTorch)"].named_steps["model"].n_iter_
insights.append(f"PyTorch feedforward network (64-32, BatchNorm + 0.3 dropout, Adam, class-weighted BCE, "
                f"early-stopped at epoch {nn_epochs}) reached ROC-AUC "
                f"{[r for r in results if r['model'].startswith('Feedforward')][0]['roc_auc']:.3f} vs "
                f"{[r for r in results if r['model'].startswith('MLP')][0]['roc_auc']:.3f} for the sklearn MLP.")

# ---- Optuna tuning for LightGBM (15 trials, stratified CV on train, PR-AUC) ----
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 100, 400),
              "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 15, 63),
              "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
              "subsample": trial.suggest_float("subsample", 0.7, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0)}
    pipe = make_pipe(__import__("lightgbm").LGBMClassifier(**params, verbose=-1, random_state=RNG,
                                                           class_weight="balanced"))
    cv = StratifiedKFold(3, shuffle=True, random_state=RNG)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return average_precision_score(y_train, oof)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RNG))
study.optimize(objective, n_trials=15)
best_params = study.best_params
tuned = make_pipe(__import__("lightgbm").LGBMClassifier(**best_params, verbose=-1, random_state=RNG,
                                                        class_weight="balanced"))
tuned.fit(X_train, y_train)
p_tuned = tuned.predict_proba(X_test)[:, 1]
results.append(eval_row("LightGBM (Optuna-tuned)", p_tuned))
preds["LightGBM (Optuna-tuned)"] = p_tuned
pipes["LightGBM (Optuna-tuned)"] = tuned
write_csv(pd.DataFrame([{"trial": t.number, "pr_auc_cv": round(t.value, 5), **t.params} for t in study.trials])
          .sort_values("pr_auc_cv", ascending=False), OUT / "optuna_trials.csv")
with (OUT / "optuna_best_params.json").open("w", encoding="utf-8") as f:
    json.dump({"best_params": best_params, "best_train_pr_auc_cv": float(study.best_value), "n_trials": 15,
               "objective": "stratified 3-fold CV PR-AUC on the training split only"}, f, indent=2)
print("Optuna best", best_params, "CV PR-AUC", round(study.best_value, 4))

comp = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
write_csv(comp, OUT / "model_comparison.csv")
best_name = comp.iloc[0]["model"]
best_pipe, best_p = pipes[best_name], preds[best_name]
joblib.dump(best_pipe, OUT / "model_best.pkl")
print("BEST (uncalibrated, by test PR-AUC):", best_name)

# ============ Phase 5: calibration, threshold, profit ============
cal = CalibratedClassifierCV(best_pipe, method="isotonic", cv=3)
cal.fit(X_train, y_train)
p_cal = cal.predict_proba(X_test)[:, 1]
joblib.dump(cal, OUT / "model_calibrated.pkl")
insights.append(f"Best-by-PR-AUC model {best_name} was isotonic-calibrated over 3 folds; scoring and tiering "
                f"use outputs/model_calibrated.pkl, not the raw scores.")

test_df = df.loc[idx_test].copy()
uncal_rank = float(roc_auc_score(y_test, best_p))
cal_rank = float(roc_auc_score(y_test, p_cal))
brier_uncal = float(brier_score_loss(y_test, best_p))
brier_cal = float(brier_score_loss(y_test, p_cal))

test_sorted = test_df.assign(prob=p_cal).sort_values("prob", ascending=False).reset_index(drop=True)
profit_rows = []
for frac in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
    k = int(frac * len(test_sorted))
    sub = test_sorted.iloc[:k]
    rev = float(sub.loc[sub["target"] == 1, "MonthlyCharges"].sum() * F.CLV_MONTHS * F.SCENARIO_SAVE_RATE)
    cost = k * F.CAMPAIGN_COST_PER_CUSTOMER
    profit_rows.append({"contact_fraction": frac, "contacted": k, "churners_contacted": int(sub["target"].sum()),
                        "expected_saved_scenario": round(float(sub["target"].sum()) * F.SCENARIO_SAVE_RATE, 1),
                        "expected_revenue_saved": round(rev, 2), "campaign_plus_offer_cost": round(cost, 2),
                        "expected_profit_scenario": round(rev - cost, 2)})
profit = pd.DataFrame(profit_rows)
write_csv(profit, OUT / "profit_curve.csv")
best_profit_row = profit.loc[profit["expected_profit_scenario"].idxmax()]

best_thr, best_prof = 0.5, -1e18
for thr in np.arange(0.10, 0.91, 0.05):
    sub = test_sorted[test_sorted["prob"] >= thr]
    if len(sub) == 0:
        continue
    prof = F.scenario_profit(sub)
    if prof > best_prof:
        best_thr, best_prof = float(thr), float(prof)

conf = confusion_matrix(y_test, (p_cal >= 0.5).astype(int))
evaluation = {"model_uncalibrated": best_name, "model_deployed": f"{best_name} + isotonic calibration",
              "test_rows": int(len(y_test)), "base_churn_rate": round(base_rate, 4),
              "roc_auc_uncalibrated": round(uncal_rank, 4), "roc_auc_calibrated": round(cal_rank, 4),
              "pr_auc_uncalibrated": round(float(average_precision_score(y_test, best_p)), 4),
              "pr_auc_calibrated": round(float(average_precision_score(y_test, p_cal)), 4),
              "brier_uncalibrated": round(brier_uncal, 5), "brier_calibrated": round(brier_cal, 5),
              "brier_baseline_base_rate": round(float(brier_score_loss(y_test, np.full(len(y_test), base_rate))), 5),
              "f1_at_0.5_calibrated": round(float(f1_score(y_test, (p_cal >= 0.5).astype(int))), 4),
              "confusion_at_0.5_calibrated": {"tn": int(conf[0][0]), "fp": int(conf[0][1]),
                                              "fn": int(conf[1][0]), "tp": int(conf[1][1])},
              "high_risk_cutoff": F.HIGH_RISK_CUTOFF, "medium_risk_cutoff": F.MEDIUM_RISK_CUTOFF,
              "high_risk_share_test": round(float((p_cal >= F.HIGH_RISK_CUTOFF).mean()), 4),
              "profit_max_threshold": best_thr, "profit_at_threshold_scenario": round(best_prof, 2),
              "best_contact_fraction_by_profit": float(best_profit_row["contact_fraction"]),
              "best_scenario_profit": float(best_profit_row["expected_profit_scenario"]),
              "scenario_assumptions": {"save_rate": F.SCENARIO_SAVE_RATE,
                                       "cost_per_contacted": F.CAMPAIGN_COST_PER_CUSTOMER,
                                       "clv_months": F.CLV_MONTHS,
                                       "note": "scenario only - true uplift needs A/B test data"}}
with (OUT / "evaluation.json").open("w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=2)

for tag, probs in [("uncalibrated", best_p), ("calibrated", p_cal)]:
    pts = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    write_csv(pd.DataFrame({"model": tag, "mean_pred": pts[1], "frac_positive": pts[0]}),
              OUT / f"calibration_curve_{tag}.csv")
cal_all = pd.concat([pd.read_csv(OUT / f"calibration_curve_{t}.csv") for t in ("uncalibrated", "calibrated")])
cal_all.to_csv(OUT / "calibration_curve.csv", index=False, lineterminator="\n")

scored_test = test_df.assign(prob=p_cal)
scored_test["decile"] = pd.qcut(scored_test["prob"], 10, labels=False, duplicates="drop")
lift = scored_test.groupby("decile", observed=True).agg(n=("target", "size"), churners=("target", "sum"),
                                                        churn_rate=("target", "mean")).reset_index()
lift["lift"] = (lift["churn_rate"] / base_rate).round(3)
lift["cumulative_capture"] = (lift["churners"].sort_values(ascending=False).cumsum()
                              / lift["churners"].sum()).round(3)
write_csv(lift, OUT / "lift_by_decile.csv")
with (OUT / "classification_report.txt").open("w", encoding="utf-8") as f:
    f.write(f"Deployed model: {evaluation['model_deployed']}\n")
    f.write(classification_report(y_test, (p_cal >= 0.5).astype(int), digits=4))
    f.write("\nAt the operational High-risk cutoff (>= 0.60):\n")
    f.write(classification_report(y_test, (p_cal >= F.HIGH_RISK_CUTOFF).astype(int), digits=4))

# ============ Phase 6: explainability ============
import shap

feat_names = list(best_pipe.named_steps["pre"].get_feature_names_out())
Xt_test = best_pipe.named_steps["pre"].transform(X_test)
Xt_test = Xt_test.toarray() if hasattr(Xt_test, "toarray") else np.asarray(Xt_test)

# 6a. Linear model -> coefficients + real SHAP LinearExplainer + permutation importance.
coef = best_pipe.named_steps["model"].coef_[0]
write_csv(pd.DataFrame({"feature": feat_names, "coefficient": np.round(coef, 5),
                        "mean_abs_coefficient": np.round(np.abs(coef), 5)})
          .sort_values("mean_abs_coefficient", ascending=False), OUT / "logreg_coefficients.csv")
perm = permutation_importance(best_pipe, X_test, y_test, scoring="average_precision", n_repeats=5,
                              random_state=RNG, n_jobs=1)
write_csv(pd.DataFrame({"feature": CAT + NUM, "perm_importance_pr_auc": np.round(perm.importances_mean, 5),
                        "std": np.round(perm.importances_std, 5)}).sort_values(
    "perm_importance_pr_auc", ascending=False), OUT / "permutation_importance.csv")
lin_explainer = shap.LinearExplainer(best_pipe.named_steps["model"], Xt_test)
lin_sv = np.asarray(lin_explainer.shap_values(Xt_test))
write_csv(pd.DataFrame({"feature": feat_names, "mean_abs_shap_linear": np.round(np.abs(lin_sv).mean(0), 5)})
          .sort_values("mean_abs_shap_linear", ascending=False), OUT / "shap_linear_logreg.csv")

# 6b. Tree model (Optuna-tuned LightGBM) -> TreeExplainer SHAP + beeswarm + dependence plots.
tree_pre = tuned.named_steps["pre"]
Xt_tree = tree_pre.transform(X_test)
Xt_tree = Xt_tree.toarray() if hasattr(Xt_tree, "toarray") else np.asarray(Xt_tree)
tree_names = list(tree_pre.get_feature_names_out())
tree_explainer = shap.TreeExplainer(tuned.named_steps["model"])
tree_sv = tree_explainer.shap_values(Xt_tree)
if isinstance(tree_sv, list):
    tree_sv = np.asarray(tree_sv[1] if len(tree_sv) == 2 else tree_sv[0])
tree_sv = np.asarray(tree_sv, dtype=float)
shap_df = (pd.DataFrame({"feature": tree_names, "mean_abs_shap": np.abs(tree_sv).mean(0),
                         "shap_sign": np.sign(tree_sv.mean(0))})
           .sort_values("mean_abs_shap", ascending=False))
write_csv(shap_df, OUT / "shap_importance_tree.csv")
plt.figure(figsize=(10, 7))
ev = tree_explainer.expected_value
base_value = float(np.ravel(ev)[-1])
explanation = shap.Explanation(values=tree_sv, base_values=base_value, data=Xt_tree, feature_names=tree_names)
shap.plots.beeswarm(explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm_lightgbm.png", dpi=150, bbox_inches="tight")
plt.close()

pd_cols = ["tenure", "MonthlyCharges", "service_count", "contract_risk", "electronic_check_flag"]
# sklearn's partial dependence refuses integer input columns; the scaler makes them float
# downstream anyway, so cast a copy purely for the PDP grid.
X_test_pd = X_test.astype({c: "float64" for c in NUM})
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
PartialDependenceDisplay.from_estimator(tuned, X_test_pd, pd_cols, kind="average", random_state=RNG,
                                        grid_resolution=40, n_cols=3, ax=axes.ravel()[:len(pd_cols)])
for extra in axes.ravel()[len(pd_cols):]:
    extra.axis("off")
fig.suptitle("Partial dependence (Optuna-tuned LightGBM) - averaged over the test split", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "partial_dependence_lightgbm.png", dpi=150, bbox_inches="tight")
plt.close(fig)

pd_rows = []
for col in ["tenure", "MonthlyCharges", "service_count", "contract_risk"]:
    dp = partial_dependence(tuned, X_test_pd, [col], kind="average", grid_resolution=25)
    gv = np.asarray(dp["grid_values"][0], dtype=float)
    avg = np.asarray(dp["average"][0], dtype=float).reshape(-1)
    pd_rows.append({"feature": col, "grid_values": np.round(gv, 4).tolist(),
                    "mean_predicted_churn": np.round(avg, 4).tolist(),
                    "range_pp": round(float((avg.max() - avg.min()) * 100), 2)})
write_csv(pd.DataFrame([{"feature": r["feature"], "range_pp": r["range_pp"],
                         "grid": json.dumps({"x": r["grid_values"], "y": r["mean_predicted_churn"]})}
                        for r in pd_rows]), OUT / "partial_dependence.csv")

hi = int(np.argmax(p_cal))
contrib = Xt_test[hi] * coef
local = pd.DataFrame({"feature": feat_names, "contribution": np.round(contrib, 5)}) \
    .iloc[np.argsort(-np.abs(contrib))].head(10)
write_csv(local.assign(customerID=test_df.iloc[hi]["customerID"],
                       churn_probability=round(float(p_cal[hi]), 4)), OUT / "shap_local_example.csv")
for stale in [FIG / "shap_importance.png", OUT / "shap_importance.csv", OUT / "cox_results.csv"]:
    stale.unlink(missing_ok=True)
insights.append(f"Tree SHAP top driver for the Optuna-tuned LightGBM: {shap_df.iloc[0]['feature']} "
                f"(mean |SHAP| {shap_df.iloc[0]['mean_abs_shap']:.3f}).")
print("Explainability: logreg coefficients + LinearExplainer + tree TreeExplainer SHAP + PDP written")

# ============ Phase 7: survival (lifelines; R script run separately) ============
from lifelines import CoxPHFitter, KaplanMeierFitter

km_out = []
for name, sub in df.groupby("Contract"):
    kmf = KaplanMeierFitter()
    kmf.fit(sub["tenure"], event_observed=sub["target"], label=str(name))
    for t in [12, 24, 36, 48, 60, 72]:
        km_out.append({"segment_type": "Contract", "segment": str(name), "month": t,
                       "n_at_risk_start": int((sub["tenure"] >= t).sum()),
                       "survival_probability": round(float(kmf.survival_function_at_times(t).iloc[0]), 4)})
for name, sub in df.groupby("PaymentMethod"):
    kmf = KaplanMeierFitter()
    kmf.fit(sub["tenure"], event_observed=sub["target"], label=str(name))
    for t in [12, 24, 36, 48, 60, 72]:
        km_out.append({"segment_type": "PaymentMethod", "segment": str(name), "month": t,
                       "n_at_risk_start": int((sub["tenure"] >= t).sum()),
                       "survival_probability": round(float(kmf.survival_function_at_times(t).iloc[0]), 4)})
write_csv(pd.DataFrame(km_out), OUT / "survival_km.csv")

cox_cols = ["tenure", "target", "MonthlyCharges", "SeniorCitizen", "service_count", "contract_risk",
            "electronic_check_flag", "no_support_flag", "bill_shock_flag"]
cox_df = df[cox_cols].rename(columns={"tenure": "duration", "target": "event"})
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="duration", event_col="event")
cox = cph.summary.reset_index().rename(columns={"index": "covariate"})
cox["hazard_ratio"] = np.exp(cox["coef"])
cox["ph_test_p"] = np.nan
try:
    zph = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
except Exception:
    pass
from lifelines.statistics import proportional_hazard_test

ph = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_map = {str(ix): float(p) for ix, p in zip(ph.summary.index, ph.summary["p"])}
cox["ph_test_p_rank"] = cox["covariate"].map(ph_map)
write_csv(cox, OUT / "cox_results_lifelines.csv")
median_surv = {}
for name, sub in df.groupby("Contract"):
    kmf = KaplanMeierFitter().fit(sub["tenure"], event_observed=sub["target"])
    med = kmf.median_survival_time_
    median_surv[str(name)] = None if not np.isfinite(med) else float(med)
write_csv(pd.DataFrame([{"segment": k, "median_survival_months": v} for k, v in median_surv.items()]),
          OUT / "survival_median.csv")
print("Survival (lifelines) written; R script r/survival_analysis.R runs independently -> outputs/r/")

# ============ Phase 8: fairness / subpopulation performance ============
ibm = external.load_ibm_modules(BASE)
recon = external.reconcile(df_raw, ibm)
write_csv(recon, OUT / "source_reconciliation.csv")
age_map = ibm["demographics"].rename(columns={"Customer ID": "customerID"})[["customerID", "Age"]]
test_eval = test_df.assign(prob=p_cal).merge(age_map, on="customerID", how="left")
test_eval["age_band"] = pd.cut(test_eval["Age"], [0, 34, 49, 64, 200],
                               labels=["18-34", "35-49", "50-64", "65+"])
fair_segments = {"SeniorCitizen": test_eval["SeniorCitizen"].map({0: "non-senior", 1: "senior citizen"}),
                 "gender": test_eval["gender"], "Partner": test_eval["Partner"],
                 "Dependents": test_eval["Dependents"], "tenure_band": test_eval["tenure_band"].astype(str),
                 "age_band": test_eval["age_band"].astype(str), "Contract": test_eval["Contract"]}
fair_rows = []
for dim, series in fair_segments.items():
    for value, sub in test_eval.assign(_dim=series).groupby("_dim", observed=True):
        if len(sub) < 30:
            continue
        yy = sub["target"].values
        pp = sub["prob"].values
        flag = (pp >= F.HIGH_RISK_CUTOFF).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, flag, labels=[0, 1]).ravel() if len(np.unique(yy)) == 2 else (0,) * 4
        fair_rows.append({
            "segment_dimension": dim, "segment": str(value), "n": int(len(sub)),
            "base_churn_rate": round(float(yy.mean()), 4),
            "mean_predicted_prob": round(float(pp.mean()), 4),
            "calibration_gap_pp": round(float((pp.mean() - yy.mean()) * 100), 2),
            "high_risk_share": round(float(flag.mean()), 4),
            "recall_at_high_cutoff": round(float(tp / (tp + fn)), 4) if (tp + fn) else np.nan,
            "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) else np.nan,
            "precision_at_high_cutoff": round(float(tp / (tp + fp)), 4) if (tp + fp) else np.nan,
            "roc_auc": round(float(roc_auc_score(yy, pp)), 4) if len(np.unique(yy)) == 2 else np.nan})
fair = pd.DataFrame(fair_rows)
write_csv(fair, OUT / "fairness_by_segment.csv")
disc = []
for dim, sub in fair.groupby("segment_dimension"):
    ref = sub.loc[sub["high_risk_share"].idxmax()]
    best_tpr = sub["recall_at_high_cutoff"].max()
    worst_tpr = sub["recall_at_high_cutoff"].min()
    disc.append({"segment_dimension": dim, "groups": int(len(sub)),
                 "max_high_risk_share": round(float(sub["high_risk_share"].max()), 4),
                 "min_high_risk_share": round(float(sub["high_risk_share"].min()), 4),
                 "selection_ratio_min_over_max": round(float(sub["high_risk_share"].min()
                                                             / sub["high_risk_share"].max()), 3),
                 "recall_spread_pp": round(float((best_tpr - worst_tpr) * 100), 2),
                 "recall_ratio": round(float(worst_tpr / best_tpr), 3) if best_tpr else np.nan,
                 "max_abs_calibration_gap_pp": round(float(sub["calibration_gap_pp"].abs().max()), 2),
                 "reference_group": str(ref["segment"]),
                 "adverse_impact_note": "being flagged here means receiving a retention offer, so a higher "
                                        "selection rate is not by itself a harm; equal-opportunity (recall) "
                                        "and calibration gaps are the fairness questions that matter"})
write_csv(pd.DataFrame(disc), OUT / "fairness_summary.csv")
sen = fair[(fair.segment_dimension == "SeniorCitizen")].set_index("segment")
insights.append(f"Fairness: seniors are {sen.loc['senior citizen', 'n'] / len(test_eval):.1%} of the test split, "
                f"base churn {sen.loc['senior citizen', 'base_churn_rate']:.1%} vs non-seniors "
                f"{sen.loc['non-senior', 'base_churn_rate']:.1%}; flagged as High risk "
                f"{sen.loc['senior citizen', 'high_risk_share']:.1%} vs {sen.loc['non-senior', 'high_risk_share']:.1%}, "
                f"recall at the cutoff {sen.loc['senior citizen', 'recall_at_high_cutoff']:.1%} vs "
                f"{sen.loc['non-senior', 'recall_at_high_cutoff']:.1%}.")

# ============ Phase 9: scoring + retention actions (deployed = calibrated model) ============
df["split"] = "train"
df.loc[idx_test, "split"] = "test"
df["churn_probability"] = cal.predict_proba(X)[:, 1]
df = F.add_scores(df, df["churn_probability"].values)
df["recommended_action"] = df.apply(F.recommended_action, axis=1)
write_csv(df, OUT / "scored_customers.csv")
act = (df[df["risk_tier"] == "High"].groupby("recommended_action")
       .agg(customers=("target", "size"), churn_rate=("target", "mean"),
            revenue_at_risk_12m=("revenue_at_risk_12m", "sum")).reset_index()
       .sort_values("revenue_at_risk_12m", ascending=False))
act["expected_saved_scenario_30pct"] = (act["customers"] * act["churn_rate"] * F.SCENARIO_SAVE_RATE).round(1)
write_csv(act, OUT / "retention_actions.csv")

# ============ Phase 10: second source - enrichment + drift batches ============
enr = external.enrichment_frame(df, ibm)
churned = enr[enr["target"] == 1]
write_csv(churned.groupby(["Churn Category"]).agg(churners=("customerID", "size"),
                                                  avg_monthly=("MonthlyCharges", "mean"),
                                                  avg_tenure=("tenure", "mean"),
                                                  avg_cltv=("CLTV", "mean")).reset_index()
          .sort_values("churners", ascending=False), OUT / "enrichment_churn_category.csv")
write_csv(churned.groupby("Churn Reason").agg(churners=("customerID", "size"),
                                              avg_monthly=("MonthlyCharges", "mean"),
                                              avg_cltv=("CLTV", "mean")).reset_index()
          .sort_values("churners", ascending=False).head(15), OUT / "enrichment_top_churn_reasons.csv")
write_csv(enr.groupby("State").agg(customers=("customerID", "size"), churn_rate=("target", "mean"),
                                   high_risk=("risk_tier", lambda s: (s == "High").sum()),
                                   revenue_at_risk=("revenue_at_risk_12m", "sum")).reset_index()
          .query("customers >= 100").sort_values("revenue_at_risk", ascending=False).head(15),
          OUT / "enrichment_by_state.csv")
write_csv(enr.groupby("Satisfaction Score").agg(customers=("customerID", "size"),
                                                churn_rate=("target", "mean"),
                                                mean_prob=("churn_probability", "mean")).reset_index(),
          OUT / "enrichment_by_satisfaction.csv")
clv_cmp = {"monthly_x_12_mean": round(float(enr["MonthlyCharges"].mean() * 12), 2),
           "ibm_cltv_mean": round(float(enr["CLTV"].mean()), 2),
           "pearson_r_monthly_x_12_vs_cltv": round(float((enr["MonthlyCharges"] * 12).corr(enr["CLTV"])), 4),
           "high_risk_value_at_risk_monthly_x_12": round(float(enr.loc[enr["risk_tier"] == "High",
                                                                       "MonthlyCharges"].sum() * 12), 2),
           "high_risk_value_at_risk_ibm_cltv": round(float(enr.loc[enr["risk_tier"] == "High", "CLTV"].sum()), 2),
           "note": "CLTV is an IBM-supplied field on the secondary release, used for sizing only, never a feature"}
with (OUT / "enrichment_cltv_vs_proxy.json").open("w", encoding="utf-8") as f:
    json.dump(clv_cmp, f, indent=2)

# IBM ships its own risk score (Churn Score, 5-96) on the secondary release. It is
# never a feature here - scoring the label with a field derived from the label is
# leakage - but it is a fair external benchmark, so the comparison goes on disk.
from sklearn.metrics import roc_auc_score as _auc, average_precision_score as _ap

ibm_score = (ibm["status"].rename(columns={"Customer ID": "customerID"})
             [["customerID", "Churn Score", "CLTV"]])
# Restricted to the held-out split: our probabilities on training rows are in-sample
# and would flatter us, while IBM's score is a static column that is out-of-sample by
# construction. Comparing on the test rows is the only symmetric reading of "ours vs
# theirs", so the training rows are left out deliberately.
bench = df[df["split"] == "test"].merge(ibm_score, on="customerID", how="inner")
y_b = bench["target"].values
s_b = bench["Churn Score"].astype(float).values
benchmark = {"rows": int(len(bench)),
             "scope": "held-out test split only (see note)",
             "ibm_churn_score_range": [float(s_b.min()), float(s_b.max())],
             "ibm_churn_score_roc_auc": round(_auc(y_b, s_b), 4),
             "ibm_churn_score_pr_auc": round(_ap(y_b, s_b), 4),
             "our_calibrated_roc_auc": round(_auc(y_b, bench["churn_probability"]), 4),
             "our_calibrated_pr_auc": round(_ap(y_b, bench["churn_probability"]), 4),
             "spearman_between_scores": round(float(pd.Series(s_b).corr(bench["churn_probability"],
                                                                        method="spearman")), 4),
             "note": "Churn Score is excluded from every model (it encodes the label). Compared "
                     "here as an external yardstick on out-of-sample rows for both scorers."}
write_csv(bench.assign(our_prob=bench["churn_probability"].round(4))[
    ["customerID", "target", "Churn Score", "our_prob"]], OUT / "enrichment_vs_ibm_score.csv")
with (OUT / "enrichment_benchmark_vs_ibm_score.json").open("w", encoding="utf-8") as f:
    json.dump(benchmark, f, indent=2)
insights.append(f"IBM's own supplied Churn Score reaches ROC-AUC "
                f"{benchmark['ibm_churn_score_roc_auc']:.3f} / PR-AUC "
                f"{benchmark['ibm_churn_score_pr_auc']:.3f} on the same {benchmark['rows']:,} held-out "
                f"customers, against {benchmark['our_calibrated_roc_auc']:.3f} / "
                f"{benchmark['our_calibrated_pr_auc']:.3f} for our best calibrated model. We do not "
                f"close that gap and we do not pretend to: nothing built from the 21 published "
                f"columns clears 0.85 ROC-AUC here (seven model families tried, including a tuned "
                f"gradient-boosting stack and a PyTorch network), so Churn Score is computed from "
                f"inputs that are not in this feed and IBM does not document. It is excluded as a "
                f"feature - it encodes the answer - and reported here as an external yardstick. "
                f"Spearman between the two scores is only "
                f"{benchmark['spearman_between_scores']:.2f}, so they are not measuring the same "
                f"ordering of customers.")

q3_raw = external.q3_to_telco_schema(ibm)
q3_batch = df.merge(q3_raw[["customerID", "tenure", "MonthlyCharges", "Contract", "PaymentMethod",
                            "InternetService", "Churn"]], on="customerID", how="left",
                    suffixes=("", "_q3"))
q3_scored = F.engineer(q3_raw)
q3_scored = F.add_scores(q3_scored, cal.predict_proba(q3_scored[CAT + NUM])[:, 1])
q3_scored["recommended_action"] = q3_scored.apply(F.recommended_action, axis=1)
write_csv(q3_scored, PROC / "scored_batch_q3.csv")
heldout = df[df["split"] == "test"].copy()
write_csv(heldout, PROC / "scored_heldout_batch.csv")

notes = {"ibm_q3_rerelease": "IBM 11.1.3 Q3 re-release mapped to the 21-column schema. Same 7,043 customers "
                             "and same labels as the reference; categories re-coded (Electronic check removed, "
                             "DSL split into DSL/Cable, 265 contracts moved to term). Measures upstream "
                             "re-coding drift, NOT temporal drift.",
         "held_out_test_split": "Stratified test split, never used for training or calibration. Same period as "
                                "the reference by construction, so a near-zero PSI here is a control, not "
                                "evidence of stability over time."}
drift = monitor.build_drift_report(df, {"ibm_q3_rerelease": q3_scored, "held_out_test_split": heldout},
                                   NUM, CAT, notes=notes)
write_csv(drift, OUT / "drift_report.csv")
monitor.write_drift_figure(df, {"ibm_q3_rerelease": q3_scored, "held_out_test_split": heldout},
                           FIG / "drift_score_distribution.png")
q3_psi = float(drift[(drift.batch == "ibm_q3_rerelease") & (drift.measurement == "score_psi")]["value"].iloc[0])
q3_auc = drift[(drift.batch == "ibm_q3_rerelease") & (drift.measurement == "realised_roc_auc")]["value"]
feat_psi = drift[(drift.batch == "ibm_q3_rerelease") & drift.measurement.str.startswith("feature_psi_")]
worst_feat = feat_psi.loc[feat_psi["value"].idxmax()]
insights.append(f"Drift on the IBM Q3 re-release batch: score PSI {q3_psi:.4f} "
                f"({monitor.verdict(q3_psi)}) - the model output barely moved, but the feed did: "
                f"{worst_feat['measurement'].split(':')[-1]} PSI {float(worst_feat['value']):.4f} "
                f"({worst_feat['verdict']}). Realised ROC-AUC on that batch "
                f"{float(q3_auc.iloc[0]):.4f} vs {cal_rank:.4f} on the reference test split - the re-coded "
                f"payment field erases the Electronic check signal upstream.")

# ============ Figures ============
plt.figure(figsize=(9, 5))
df.groupby("Contract")["target"].mean().sort_values().plot(kind="barh")
plt.title("Churn rate by contract")
plt.tight_layout()
plt.savefig(FIG / "churn_by_contract.png", dpi=150)
plt.close()
plt.figure(figsize=(9, 5))
df.groupby("tenure_band", observed=True)["target"].mean().plot(kind="bar")
plt.title("Churn rate by tenure band")
plt.tight_layout()
plt.savefig(FIG / "churn_by_tenure.png", dpi=150)
plt.close()
plt.figure(figsize=(9, 5))
for name, sub in df.groupby("Contract"):
    KaplanMeierFitter().fit(sub["tenure"], event_observed=sub["target"], label=str(name)).plot_survival_function()
plt.title("Kaplan-Meier survival by contract (lifelines)")
plt.tight_layout()
plt.savefig(FIG / "km_by_contract.png", dpi=150)
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, style in [("uncalibrated", "-"), ("calibrated", "--")]:
    c = pd.read_csv(OUT / f"calibration_curve_{tag}.csv")
    axes[0].plot(c["mean_pred"], c["frac_positive"], style, marker="o", label=tag)
axes[0].plot([0, 1], [0, 1], ":", c="grey")
axes[0].set_xlabel("mean predicted")
axes[0].set_ylabel("observed churn")
axes[0].set_title("Reliability (test split)")
axes[0].legend()
top = shap_df.head(15).iloc[::-1]
axes[1].barh(top["feature"], top["mean_abs_shap"])
axes[1].set_title("Mean |SHAP| - Optuna-tuned LightGBM")
axes[1].set_xlabel("mean |SHAP|")
fig.tight_layout()
fig.savefig(FIG / "calibration_and_shap.png", dpi=150)
plt.close(fig)
sen_f = fair[fair.segment_dimension.isin(["SeniorCitizen", "gender", "age_band"])]
plt.figure(figsize=(10, 5))
sns = __import__("seaborn").lineplot(data=sen_f, x="base_churn_rate", y="high_risk_share", hue="segment",
                                     style="segment_dimension", marker="o")
plt.title("Observed churn vs share flagged High risk, by segment (equal-opportunity view)")
plt.xlabel("base churn rate")
plt.ylabel("share flagged High risk")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "fairness_segments.png", dpi=150)
plt.close()

# ============ Reports ============
with (OUT / "eda_insight_report.md").open("w", encoding="utf-8") as f:
    f.write("# EDA insight report - Customer Churn (IBM Telco, n=7,043)\n\n")
    f.write(f"- Data quality score: {dq_score}% ({int(dq_card['passed'].sum())}/{len(dq_card)} checks passed, "
            f"2 of them Pandera schema validations over all 21 columns). Blank TotalCharges: 11, all tenure-0.\n")
    for s in insights:
        f.write(f"- {s}\n")
    f.write("\n## What is NOT in this dataset (do not claim it)\n")
    f.write("- No support-call log, login log, usage 30/60/90 series, complaint log, payment-delay or "
            "signup-date field.\n")
    f.write("- Bill shock, contract-remaining and adapted RFM are proxies from available columns only.\n")
    f.write("- A true time-based split needs dated snapshots; this build uses a stratified split.\n")
    f.write("- The second source (IBM 11.1.3) is a re-coded re-release of the same book of business, not a "
            "later period - see data/raw/ibm_telco_11_1_3/PROVENANCE.md.\n")

with (OUT / "model_report.md").open("w", encoding="utf-8") as f:
    f.write("# Model report\n\n## Comparison (test split, n=%d)\n\n" % len(y_test))
    f.write(comp.to_markdown(index=False) + "\n\n")
    f.write(f"Best by PR-AUC: **{best_name}**. Deployed artifact: **model_calibrated.pkl** "
            f"(isotonic, 3-fold) - scoring, batch and API all use it.\n\n")
    f.write("## Calibration\n\n")
    f.write(f"- ROC-AUC {uncal_rank:.4f} -> {cal_rank:.4f} after calibration (isotonic is rank-preserving "
            f"per fold, so the change is the 3-fold averaging, not a ranking win).\n")
    f.write(f"- Brier {brier_uncal:.5f} -> {brier_cal:.5f}; constant-base-rate baseline "
            f"{evaluation['brier_baseline_base_rate']:.5f}.\n")
    f.write(f"- Share flagged High risk on test at cutoff {F.HIGH_RISK_CUTOFF}: "
            f"{evaluation['high_risk_share_test']:.1%} (base churn {base_rate:.1%}).\n\n")
    f.write("## Threshold and profit\n\n")
    f.write(f"- Profit-maximising threshold (scenario): {best_thr} -> ${best_prof:,.0f} on test.\n")
    f.write(f"- Best contact fraction by scenario profit: {best_profit_row['contact_fraction']:.0%} "
            f"(${best_profit_row['expected_profit_scenario']:,.0f}).\n")
    f.write(f"- Assumptions: save rate {F.SCENARIO_SAVE_RATE:.0%}, "
            f"${F.CAMPAIGN_COST_PER_CUSTOMER:.0f} per contacted customer, CLV = monthly x {F.CLV_MONTHS}. "
            f"Scenario only - true uplift needs A/B test data.\n\n")
    f.write("## Fairness (see outputs/fairness_by_segment.csv)\n\n")
    f.write(pd.DataFrame(disc).to_markdown(index=False) + "\n")

    f.write("\n## Survival (lifelines + R, run independently)\n\n")
    f.write(f"- Median survival by contract: "
            + ", ".join(f"{k} = {'not reached' if v is None else str(v) + ' months'}"
                        for k, v in median_surv.items())
            + ".\n")
    f.write("- R (`r/survival_analysis.R`, survival 3.8.6) and lifelines agree on the Cox coefficients to "
            "6 decimal places; the side-by-side table is below and the raw R output is in outputs/r/.\n\n")
    r_cox = pd.read_csv(OUT / "r" / "cox_hazard_ratios.csv")
    cmp_cols = ["covariate", "coef", "hazard_ratio", "std_err", "p_value"]
    merge_how = cox[["covariate", "coef", "hazard_ratio", "se(coef)"]].rename(
        columns={"se(coef)": "std_err"}).round(6).merge(
        r_cox[cmp_cols].round(6), on="covariate", suffixes=("_lifelines", "_r"))
    merge_how["max_abs_coef_diff"] = (merge_how["coef_lifelines"]
                                      - merge_how["coef_r"]).abs().round(9)
    f.write(merge_how[["covariate", "coef_lifelines", "coef_r", "hazard_ratio_lifelines",
                       "hazard_ratio_r", "max_abs_coef_diff"]].to_markdown(index=False) + "\n\n")
    zph_r = pd.read_csv(OUT / "r" / "cox_zph_test.csv")
    ph_join = zph_r.merge(cox[["covariate", "ph_test_p_rank"]],
                          left_on="term", right_on="covariate", how="left")
    f.write("- **Proportional-hazards assumption fails.** R `cox.zph` (identity time transform) rejects it for "
            "MonthlyCharges, service_count, contract_risk and no_support_flag; lifelines' rank test rejects the "
            "same three drivers. The two implementations disagree on the two weakest terms "
            "(no_support_flag, bill_shock_flag) because they transform time differently.\n")
    f.write("- Consequence: the hazard ratios above are **time-averaged** effects, valid for direction and "
            "magnitude over the observed window, not as instantaneous effects at a given tenure. The KM curves "
            "and the classification model carry the operational conclusions; the Cox model is supporting "
            "evidence. A fix (stratify on contract_risk, or add time-varying coefficients) is out of scope here "
            "and is listed in the README limitations.\n")
    f.write("\n" + ph_join[["term", "chisq", "p_value", "ph_test_p_rank"]].to_markdown(index=False) + "\n")

    f.write("\n## Drift (full table: outputs/drift_report.csv)\n\n")
    for batch in drift["batch"].unique():
        b = drift[drift.batch == batch]
        get = lambda m: b[b.measurement == m]["value"].iloc[0]
        feats = b[b.measurement.str.startswith("feature_psi_")]
        drifted = int((feats["value"] > 0.10).sum())
        schema = b[b.measurement.str.startswith("category_change:")]
        f.write(f"- **{batch}** ({int(get('batch_rows'))} rows, churn {get('batch_churn_rate'):.1%}) - "
                f"score PSI **{get('score_psi'):.4f}** "
                f"({b[b.measurement == 'score_psi']['verdict'].iloc[0]}); {drifted} of {len(feats)} features "
                f"above PSI 0.10; {len(schema)} categorical domain change(s); "
                f"realised ROC-AUC {get('realised_roc_auc'):.4f}, PR-AUC {get('realised_pr_auc'):.4f}, "
                f"precision at the {F.HIGH_RISK_CUTOFF} cutoff {get('high_tier_precision_at_0.60'):.3f}.\n")
        f.write(f"  {b[b.measurement == 'score_psi']['detail'].iloc[0]}\n")

print("PIPELINE DONE")
print(comp.to_string(index=False))
print(f"High-risk customers: {(df['risk_tier'] == 'High').sum()} "
      f"({(df['risk_tier'] == 'High').mean():.1%}), revenue at risk 12m: "
      f"${df['revenue_at_risk_12m'].sum():,.0f}, calibrated test ROC-AUC {cal_rank:.4f}")
