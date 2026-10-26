"""Generate customer_churn_retention_platform.ipynb, then execute it with nbconvert.

The notebook is a narrative over the SAME modules the pipeline runs (features.py, dq.py,
monitor.py, external.py, torch_nn.py), not a second implementation of them. That is the
whole point: the version this file replaced re-typed engineer() and the model zoo inline,
so it silently drifted to "5 models, no calibration, no survival" while the pipeline had
moved on. Cells read committed artefacts for comparison and write nothing.

    python src/make_notebook.py            # regenerate the .ipynb (unexecuted)
    python src/make_notebook.py --execute  # regenerate and run it end to end
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

BASE = Path(__file__).resolve().parents[1]
NB_PATH = BASE / "customer_churn_retention_platform.ipynb"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip()))


def code(text: str) -> None:
    cells.append(("code", text.strip()))


# --------------------------------------------------------------------------- front matter
md("""
# Customer Churn Prediction and Retention Analytics

**Business problem.** A telecom operator loses subscribers while retention spend goes out
flat to everyone. This notebook answers four questions with one model contract: who will
churn, what is driving it, when a customer is likely to leave, and which customers are
worth paying to keep.

**Stack.** Python for the whole ML path (pandas, scikit-learn, XGBoost, LightGBM, PyTorch,
Optuna, SHAP, lifelines, Pandera, DuckDB, FastAPI, Streamlit) and R for an independent
survival cross-check (`r/survival_analysis.R`).

**How to read this notebook.** Every cell that does real work calls the same functions the
production pipeline calls (`src/features.py`, `src/dq.py`, `src/monitor.py`,
`src/external.py`). It writes no files. The last cell asserts that what it computed
matches the committed artefacts in `outputs/`, so a reader can tell whether the notebook
and the pipeline agree rather than taking that on faith.

**Data.** IBM Telco Customer Churn (7,043 rows, 21 columns) as the primary feed, plus
IBM's *Telco customer churn (11.1.3+)* six-module release as a second source. The second
release is a re-coded re-release of the same book of business, not a later period -
see `data/raw/ibm_telco_11_1_3/PROVENANCE.md` and the Drift section.
""")

code("""
%matplotlib inline
import json, subprocess, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9, 4.5)

import features as F
import dq
import monitor
import external

RNG = 42
OUT = Path("outputs")
pd.set_option("display.width", 160)
print("pandas", pd.__version__, "| feature contract:", len(F.CAT), "categorical +", len(F.NUM), "numeric")
""")

# --------------------------------------------------------------------------- 1. ingestion + DQ
md("""
## 1. Ingestion and data quality

The CSV is read with `dtype=str, keep_default_na=False` on purpose. `TotalCharges` is the
one column with legitimate blanks - 11 rows, all `tenure = 0`, customers who have not
billed yet - and the default reader decides what a blank means instead of the pipeline.
Pandera schemas (raw and typed) plus 35 business checks produce the scorecard.
""")

code("""
raw = F.load_raw("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
raw.shape, list(raw.columns) == dq.EXPECTED_COLUMNS
""")

code("""
for name, schema in [("telco_raw", dq.telco_raw_schema()),
                     ("telco_clean", dq.telco_clean_schema())]:
    frame = raw if name == "telco_raw" else F.coerce_numeric(raw)
    print(name, dq.validate_schema(frame, schema))
""")

code("""
card = dq.run_dq_checks(raw)
print(f"DQ score {card['passed'].mean():.1%}  ({int(card['passed'].sum())}/{len(card)} checks)")
failures = card.loc[~card["passed"], ["check", "layer", "detail"]]
failures if len(failures) else "all checks green"
""")

code("""
blanks = raw[raw["TotalCharges"].str.strip() == ""]
blanks[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]
""")

code("""
df = F.engineer(raw)
df["target"].mean(), df["tenure"].min(), df["tenure"].max(), df["MonthlyCharges"].max()
""")

md("""
The domain values are not incidental: they are what `dq.ALLOWED` and the FastAPI request
model validate against, so a category that is not on this list cannot reach the encoder
where `handle_unknown="ignore"` would silently zero it.
""")

code("""
pd.DataFrame({"column": dq.EXPECTED_COLUMNS,
              "distinct": [raw[c].nunique() for c in dq.EXPECTED_COLUMNS],
              "levels": [sorted(raw[c].unique())[:4] for c in dq.EXPECTED_COLUMNS]}).head(21)
""")

# --------------------------------------------------------------------------- 2. EDA
md("""
## 2. Where the churn actually is

Three structural facts drive everything downstream, and they are visible before any model
is fitted: month-to-month contracts churn at roughly three times the two-year rate, the
first twelve months are where customers are lost, and no online security or no tech
support is a stronger signal than price.
""")

code("""
def churn_by(col):
    t = df.groupby(col, observed=True)["target"].agg(churners="sum", customers="size", churn_rate="mean")
    return t.sort_values("churn_rate", ascending=False).round(4)

churn_by("Contract")
""")

code("""
churn_by("tenure_band")
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
churn_by("Contract")[::-1].plot(kind="barh", y="churn_rate", ax=axes[0], legend=False,
                                title="Churn rate by contract")
churn_by("tenure_band").plot(kind="bar", y="churn_rate", ax=axes[1], legend=False,
                             title="Churn rate by tenure band")
for ax in axes:
    ax.set_ylabel("churn rate")
plt.tight_layout(); plt.show()
""")

code("""
for col in ["PaymentMethod", "InternetService", "OnlineSecurity", "TechSupport", "SeniorCitizen"]:
    print(col); print(churn_by(col).to_string()); print()
""")

code("""
from scipy.stats import chi2_contingency, mannwhitneyu

tests = []
for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity",
            "PaperlessBilling", "MultipleLines"]:
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[col], df["target"]))
    tests.append({"test": "chi-square", "variable": col, "statistic": round(chi2, 1), "p_value": f"{p:.3e}"})
for col in ["tenure", "MonthlyCharges", "TotalCharges", "service_count"]:
    u, p = mannwhitneyu(df.loc[df["target"] == 1, col], df.loc[df["target"] == 0, col])
    tests.append({"test": "Mann-Whitney U", "variable": col, "statistic": round(u, 1), "p_value": f"{p:.3e}"})
stat_tests = pd.DataFrame(tests)
stat_tests
""")

md("""
Every test rejects at p &lt; 0.001, which is the honest framing: with 7,043 rows association
is not the interesting part. Effect size is, and it is in the tables above.
""")

# --------------------------------------------------------------------------- 3. segmentation
md("""
## 3. Segmentation on a snapshot

There is no signup date, so RFM here is adapted rather than literal: **R** is the tenure
quintile (a proxy for recency, not days-since-last-activity), **F** is the count of
subscribed services, **M** is monthly charges. Stating the substitution matters more than
the score.
""")

code("""
seg = df.copy()
for src, out in [("tenure", "r_score"), ("service_count", "f_score"), ("MonthlyCharges", "m_score")]:
    seg[out] = pd.qcut(seg[src], 5, labels=False, duplicates="drop") + 1
seg["rfm_total"] = seg["r_score"] + seg["f_score"] + seg["m_score"]
bands = pd.cut(seg["rfm_total"], [0, 6, 8, 10, 12, 15],
               labels=["At risk", "Mid value", "Growing", "Core", "Champions"])
seg.assign(band=bands).groupby("band", observed=True).agg(
    customers=("target", "size"), churn_rate=("target", "mean"),
    monthly_revenue=("MonthlyCharges", "sum")).round(3)
""")

# --------------------------------------------------------------------------- 4. modelling
md("""
## 4. Modelling

`train_test_split` with `stratify=y` and a fixed seed. A time-based split is not available
on a snapshot and pretending otherwise would be the worst kind of leak, so the limitation
is recorded instead. The model input contract is `F.CAT + F.NUM`, built by
`F.build_preprocessor()` - one-hot with `handle_unknown="ignore"` plus a standard scaler.
""")

code("""
from sklearn.model_selection import train_test_split

X, y = df[F.CAT + F.NUM], df["target"].values
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=RNG, stratify=y)
pre = F.build_preprocessor()
print(len(X_train), "train /", len(X_test), "test | base churn rate", round(y.mean(), 4))
""")

code("""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, brier_score_loss,
                             confusion_matrix, precision_score)
import xgboost, lightgbm
from torch_nn import TorchFeedforwardClassifier

def make_pipe(mdl):
    return Pipeline([("pre", pre), ("model", mdl)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG,
                                            class_weight="balanced_subsample", n_jobs=-1),
    "XGBoost": xgboost.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                     n_jobs=-1, random_state=RNG),
    "LightGBM": lightgbm.LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05,
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
""")

md("""
The PyTorch network is a real feedforward net, not a wrapper around sklearn's MLP:
64-32 with BatchNorm, dropout 0.3, Adam on a class-weighted BCE loss, mini-batch
training with early stopping on validation loss. It is exposed as an sklearn estimator so
it enters the same `Pipeline` and the same comparison table as everything else.
""")

code("""
results, preds, pipes = [], {}, {}
for name, mdl in models.items():
    pipe = make_pipe(mdl)
    pipe.fit(X_train, y_train)
    pipes[name] = pipe
    preds[name] = pipe.predict_proba(X_test)[:, 1]
    results.append(eval_row(name, preds[name]))
    print(f"  {name}: ROC-AUC {results[-1]['roc_auc']:.4f}")

torch_epochs = pipes["Feedforward NN (PyTorch)"].named_steps["model"].n_iter_
print("PyTorch net early-stopped at epoch", torch_epochs)
""")

# --------------------------------------------------------------------------- 5. optuna
md("""
## 5. Optuna tuning

15 TPE trials, seeded at 42. The objective is **stratified 3-fold CV PR-AUC on the
training split only** - never test-set score, which is how tuned models quietly become
overfit to the evaluation you were supposed to keep clean.
""")

code("""
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 100, 400),
              "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 15, 63),
              "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
              "subsample": trial.suggest_float("subsample", 0.7, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0)}
    pipe = make_pipe(lightgbm.LGBMClassifier(**params, verbose=-1, random_state=RNG, class_weight="balanced"))
    cv = StratifiedKFold(3, shuffle=True, random_state=RNG)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return average_precision_score(y_train, oof)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RNG))
study.optimize(objective, n_trials=15)
best_params = study.best_params
trials = (pd.DataFrame([{"trial": t.number, "pr_auc_cv": round(t.value, 5), **t.params}
                        for t in study.trials]).sort_values("pr_auc_cv", ascending=False))
print("best CV PR-AUC", round(study.best_value, 4))
trials.head(6)
""")

code("""
tuned = make_pipe(lightgbm.LGBMClassifier(**best_params, verbose=-1, random_state=RNG, class_weight="balanced"))
tuned.fit(X_train, y_train)
preds["LightGBM (Optuna-tuned)"] = tuned.predict_proba(X_test)[:, 1]
pipes["LightGBM (Optuna-tuned)"] = tuned
results.append(eval_row("LightGBM (Optuna-tuned)", preds["LightGBM (Optuna-tuned)"]))

comp = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
comp
""")

md("""
Tuning LightGBM does not beat the regularised logistic regression on this dataset, and
that is reported rather than hidden. Seven model families were tried (linear, bagged,
two gradient-boosting stacks, sklearn MLP, the PyTorch network, tuned LightGBM); nothing
built from the 21 published columns clears ROC-AUC 0.85.
""")

# --------------------------------------------------------------------------- 6. calibration
md("""
## 6. Calibration and thresholds

The best model by test PR-AUC is isotonic-calibrated with `CalibratedClassifierCV(cv=3)`
and that **calibrated** artefact is what scores every customer, what the batch job loads
and what the API serves. Ranking barely moves (isotonic is monotone); the probability
scale does, which is the whole reason to bother - the tier cutoffs and the value-at-risk
arithmetic are read as probabilities.
""")

code("""
best_name = comp.iloc[0]["model"]
best_pipe, best_p = pipes[best_name], preds[best_name]
cal = CalibratedClassifierCV(best_pipe, method="isotonic", cv=3)
cal.fit(X_train, y_train)
p_cal = cal.predict_proba(X_test)[:, 1]

print(f"deployed model: {best_name} + isotonic calibration over 3 folds")
print(f"ROC-AUC  uncalibrated {roc_auc_score(y_test, best_p):.4f}   calibrated {roc_auc_score(y_test, p_cal):.4f}")
print(f"PR-AUC   uncalibrated {average_precision_score(y_test, best_p):.4f}   calibrated {average_precision_score(y_test, p_cal):.4f}")
print(f"Brier    uncalibrated {brier_score_loss(y_test, best_p):.5f}   calibrated {brier_score_loss(y_test, p_cal):.5f}"
      f"   base-rate-only {brier_score_loss(y_test, np.full(len(y_test), y.mean())):.5f}")
print(confusion_matrix(y_test, (p_cal >= 0.5).astype(int)))
""")

code("""
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (label, p) in zip(axes[:2], [("uncalibrated", best_p), ("isotonic-calibrated", p_cal)]):
    fx, fy = calibration_curve(y_test, p, n_bins=10, strategy="uniform")
    ax.plot(fx, fy, marker="o", label=label)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="perfect")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed churn rate")
    ax.set_title(f"reliability - {label}"); ax.legend(fontsize=8)
order = np.argsort(-p_cal)
top = order[: max(1, int(0.10 * len(p_cal)))]
axes[2].bar(["base rate", "top decile"], [y_test.mean(), y_test[top].mean()], color=["grey", "crimson"])
axes[2].set_title("top-decile lift"); axes[2].set_ylabel("churn rate")
plt.tight_layout(); plt.show()
""")

md("""
### What is a saved customer worth?

Scenario arithmetic, stated as a scenario: 30% save rate on contacted churners, \$40 per
contacted customer (campaign \$10 + offer \$30), CLV = monthly charges x 12. Real uplift
needs an A/B test, and this dataset does not contain one.
""")

code("""
test_df = df.loc[idx_test]
ranked = test_df.assign(prob=p_cal).sort_values("prob", ascending=False).reset_index(drop=True)
rows = []
for frac in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
    k = int(frac * len(ranked))
    sub = ranked.iloc[:k]
    rev = float(sub.loc[sub["target"] == 1, "MonthlyCharges"].sum() * F.CLV_MONTHS * F.SCENARIO_SAVE_RATE)
    cost = k * F.CAMPAIGN_COST_PER_CUSTOMER
    rows.append({"contact_fraction": frac, "contacted": k, "churners_contacted": int(sub["target"].sum()),
                 "revenue_saved_scenario": round(rev, 2), "contact_cost": round(cost, 2),
                 "profit_scenario": round(rev - cost, 2)})
profit = pd.DataFrame(rows)
profit
""")

code("""
thrs = list(np.arange(0.10, 0.91, 0.05))
curve = pd.DataFrame({"threshold": [round(t, 2) for t in thrs],
                      "profit": [F.scenario_profit(ranked[ranked["prob"] >= t]) if (ranked["prob"] >= t).any() else 0.0
                                 for t in thrs]})
best_thr_row = curve.loc[curve["profit"].idxmax()]
print("best threshold by scenario profit:", best_thr_row["threshold"], "->",
      round(float(best_thr_row["profit"]), 2))
curve.head(8)
""")

# --------------------------------------------------------------------------- 7. explainability
md("""
## 7. Explainability

Four views of the same model, because they answer different questions: SHAP beeswarm for
direction and magnitude on the tree model, partial dependence for the shape of one
feature's effect, signed standardised coefficients for the linear model, and permutation
importance for how much the model *loses* without a feature.
""")

code("""
import shap

tree_model = pipes["LightGBM (Optuna-tuned)"]
tree_pre = tree_model.named_steps["pre"]
Xt_tree = tree_pre.transform(X_test)
Xt_tree = Xt_tree.toarray() if hasattr(Xt_tree, "toarray") else np.asarray(Xt_tree)
tree_names = list(tree_pre.get_feature_names_out())
tree_explainer = shap.TreeExplainer(tree_model.named_steps["model"])
tree_sv = tree_explainer.shap_values(Xt_tree)
if isinstance(tree_sv, list):
    tree_sv = np.asarray(tree_sv[1] if len(tree_sv) == 2 else tree_sv[0])
tree_sv = np.asarray(tree_sv, dtype=float)
shap.plots.beeswarm(shap.Explanation(tree_sv, base_values=float(np.ravel(tree_explainer.expected_value)[-1]),
                                     data=Xt_tree, feature_names=tree_names), max_display=15)
(pd.Series(np.abs(tree_sv).mean(0), index=tree_names, name="mean_abs_shap")
 .sort_values(ascending=False).head(10))
""")

code("""
lgbm = lightgbm.LGBMClassifier(**best_params, verbose=-1, random_state=RNG, class_weight="balanced")
pipe_for_pdp = make_pipe(lgbm).fit(X_train, y_train)
from sklearn.inspection import partial_dependence, permutation_importance

# sklearn's partial dependence refuses integer input columns; StandardScaler makes them
# float downstream anyway, so a float copy of the numeric block is all this needs.
X_test_pd = X_test.astype({c: "float64" for c in F.NUM})
grid = ["tenure", "MonthlyCharges", "contract_risk", "service_count", "no_support_flag"]
means = {}
for col in grid:
    pd_res = partial_dependence(pipe_for_pdp, X_test_pd, [col], kind="average", grid_resolution=12)
    means[col] = float(pd_res["average"][0][-1] - pd_res["average"][0][0])
    plt.plot(pd_res["grid_values"][0], pd_res["average"][0], marker="o", label=col)
plt.axhline(y_test.mean(), ls="--", c="grey", label="test churn rate")
plt.xlabel("feature value"); plt.ylabel("mean predicted churn probability")
plt.title("Partial dependence (isolate, not joint)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()
pd.Series(means, name="pdp_end_to_end_delta").sort_values(ascending=False)
""")

code("""
perm = permutation_importance(tree_model, X_test, y_test, scoring="average_precision",
                              n_repeats=8, random_state=RNG, n_jobs=1)
pd.DataFrame({"feature": X_test.columns, "pr_auc_drop": np.round(perm.importances_mean, 5),
              "std": np.round(perm.importances_std, 5)}) \\
    .sort_values("pr_auc_drop", ascending=False).head(10)
""")

md("""
The linear model is the deployed one, so its coefficients are worth reading directly. They
cover the 45 one-hot levels plus the 16 standardised numeric columns - log-odds per level
and per standard deviation respectively. The sign and the ranking agree with SHAP; where
they do not (a coefficient that looks large but costs nothing under permutation) the
permutation table is the tie-breaker.
""")

code("""
lr = pipes["Logistic Regression"]
feat_names = list(lr.named_steps["pre"].get_feature_names_out())
coefs = pd.Series(lr.named_steps["model"].coef_[0], index=feat_names,
                  name="std_log_odds").sort_values()
print(coefs.head(6).to_string()); print("..."); print(coefs.tail(6).to_string())
""")

code("""
lin_expl = shap.LinearExplainer(lr.named_steps["model"], lr.named_steps["pre"].transform(X_train))
Xt_lr = lr.named_steps["pre"].transform(X_test)
Xt_lr = Xt_lr.toarray() if hasattr(Xt_lr, "toarray") else np.asarray(Xt_lr)
lin_sv = np.asarray(lin_expl.shap_values(Xt_lr), dtype=float)
(pd.DataFrame({"feature": feat_names, "mean_abs_shap": np.abs(lin_sv).mean(axis=0)})
 .sort_values("mean_abs_shap", ascending=False).head(10))
""")

# --------------------------------------------------------------------------- 8. survival
md("""
## 8. Survival: when customers leave

Telco churn data is a snapshot. `tenure` is time observed at the snapshot and `Churn` is
the status at that moment, so a Kaplan-Meier curve here is a survival-to-date estimate,
not a time-to-event series built from signup dates. `Contract` is the contract observed at
the snapshot, not the contract signed - which is why the KM split by contract is
descriptive and not a causal contract effect.

lifelines runs here; R's `survival` package runs independently in `r/survival_analysis.R`
and writes its own tables to `outputs/r/`. Two engines, same coefficients, is the point.
""")

code("""
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(9, 4.5))
for band, sub in df.groupby("Contract"):
    kmf.fit(sub["tenure"], sub["target"], label=f"{band} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, ci_show=False)
ax.set_xlabel("tenure in months (observed at snapshot)"); ax.set_ylabel("S(t) = still a customer")
ax.set_title("Kaplan-Meier by contract - descriptive, contract observed at snapshot")
plt.tight_layout(); plt.show()
""")

code("""
mm = df[df["Contract"] == "Month-to-month"]; two = df[df["Contract"] == "Two year"]
lr_test = logrank_test(mm["tenure"], two["tenure"], mm["target"], two["target"])
km_all = KaplanMeierFitter().fit(df["tenure"], df["target"])
print(f"log-rank month-to-month vs two-year: chi2 {lr_test.test_statistic:.1f}, p {lr_test.p_value:.3e}")
med = km_all.median_survival_time_
print("median survival:", f"not reached - S(t) stays above 0.5 because only {df['target'].mean():.1%} "
                          "of this book churned" if not np.isfinite(med) else f"{med:.0f} months")
""")

code("""
# Same specification as src/run_pipeline.py Phase 7 and r/survival_analysis.R: no penaliser,
# tenure as the duration, Churn as the event. Deviating here would break the R cross-check.
cox_cols = ["tenure", "target", "MonthlyCharges", "SeniorCitizen", "service_count", "contract_risk",
            "electronic_check_flag", "no_support_flag", "bill_shock_flag"]
cox_df = df[cox_cols].rename(columns={"tenure": "duration", "target": "event"})
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="duration", event_col="event")
cph.print_summary()
""")

code("""
hr = pd.DataFrame({"term": cph.params_.index, "hazard_ratio": np.exp(cph.params_.values),
                   "p_value": cph.summary["p"].values})
ph = proportional_hazard_test(cph, cox_df, time_transform="rank")
hr["ph_test_p_rank"] = hr["term"].map({str(i): float(p) for i, p in zip(ph.summary.index, ph.summary["p"])})
hr
""")

md("""
### The R cross-check

`r/survival_analysis.R` fits the same specification with R's `survival` 3.8.6 (`coxph`)
and writes `outputs/r/cox_hazard_ratios.csv`, the KM tables, the log-rank results and the
Schoenfeld `cox.zph` test. Joining the two engines on the term name should give zero
difference.
""")

code("""
r_cox = pd.read_csv("outputs/r/cox_hazard_ratios.csv")
py_terms = hr.set_index("term")["hazard_ratio"]
r_terms = r_cox.set_index("covariate")["hazard_ratio"]
common = py_terms.index.intersection(r_terms.index)
cmp2 = pd.DataFrame({"lifelines": py_terms[common].round(6), "R_coxph": r_terms[common].round(6)})
cmp2["abs_diff"] = (cmp2["lifelines"] - cmp2["R_coxph"]).abs()
print("max absolute hazard-ratio difference:", float(cmp2["abs_diff"].max()))
cmp2
""")

code("""
zph = pd.read_csv("outputs/r/cox_zph_test.csv")
zph
""")

md("""
The proportional-hazards assumption is **violated** for `MonthlyCharges`, `service_count`
and `contract_risk` in both engines (lifelines' rank test additionally rejects
`bill_shock_flag`; the two disagree on the weakest terms because R's `cox.zph` and
lifelines use different time transforms). The hazard ratios are therefore reported as
*time-averaged* effects over the observed window, and the operational conclusions come
from the KM curves and the classifier, not from a Cox p-value.
""")

# --------------------------------------------------------------------------- 9. fairness
md("""
## 9. Fairness and subpopulation performance

A single ROC-AUC hides who the model is bad at. These are the standard subgroup metrics at
the deployed 0.60 cutoff - selection rate, recall, false-positive rate, precision, the
calibration gap and adverse impact - computed for segments the business actually acts on.
`SeniorCitizen`, `gender` and `Dependents` are the closest thing to protected attributes
this dataset carries; it has no race, ethnicity or national-origin field, so the check is
partial and labelled as such.
""")

code("""
test_scored = F.add_scores(df.loc[idx_test], p_cal)

def segment_metrics(frame, col):
    rows = []
    for value, g in frame.groupby(col, observed=True):
        flag = (g["churn_probability"] >= F.HIGH_RISK_CUTOFF).astype(int)
        tp = int(((flag == 1) & (g["target"] == 1)).sum()); fp = int(((flag == 1) & (g["target"] == 0)).sum())
        fn = int(((flag == 0) & (g["target"] == 1)).sum()); n0 = int((g["target"] == 0).sum())
        rows.append({"dimension": col, "segment": str(value), "n": len(g),
                     "base_churn_rate": round(float(g["target"].mean()), 4),
                     "mean_predicted_prob": round(float(g["churn_probability"].mean()), 4),
                     "calibration_gap_pp": round(float((g["churn_probability"].mean() - g["target"].mean()) * 100), 2),
                     "high_risk_share": round(float(flag.mean()), 4),
                     "recall_at_high_cutoff": round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
                     "false_positive_rate": round(fp / n0, 4) if n0 else np.nan,
                     "precision_at_high_cutoff": round(tp / (tp + fp), 4) if (tp + fp) else np.nan,
                     "roc_auc": round(float(roc_auc_score(g["target"], g["churn_probability"])), 4)
                                if g["target"].nunique() > 1 else np.nan})
    return pd.DataFrame(rows)

fair = pd.concat([segment_metrics(test_scored, c) for c in
                  ["SeniorCitizen", "gender", "Dependents", "Partner", "Contract", "InternetService",
                   "PaymentMethod", "tenure_band"]], ignore_index=True)
fair
""")

code("""
senior = fair[fair["dimension"] == "SeniorCitizen"].set_index("segment")
adv = senior.loc["1", "high_risk_share"] / senior.loc["0", "high_risk_share"]
print(f"adverse impact (senior / non-senior High-tier share): {adv:.3f}  - four-fifths rule floor is 0.80")
print(f"recall gap: {senior.loc['1','recall_at_high_cutoff'] - senior.loc['0','recall_at_high_cutoff']:+.4f}")
print(f"calibration gap (pp): senior {senior.loc['1','calibration_gap_pp']:+.2f} vs "
      f"non-senior {senior.loc['0','calibration_gap_pp']:+.2f}")
""")

md("""
Read the calibration column before the AUC column. A segment where mean predicted
probability sits above observed churn is a segment where the offer list over-fires; below
means the model under-recognises that group's risk and the retention budget never reaches
it. Adverse impact on the *selection* rate is the number an auditor asks for, and it is
deliberately reported alongside the base-rate difference, because the two segments churn at
materially different rates in the data itself.
""")

# --------------------------------------------------------------------------- 10. scoring
md("""
## 10. Scoring and the retention list

Deployed artefact: the calibrated model. Tiering, value at risk and the action rule all
come from `features.py`, which is the only copy of that logic in the repo - the batch job,
the FastAPI endpoint, the SQL mirror and the dashboard all read the same contract.
""")

code("""
df["split"] = "train"
df.loc[idx_test, "split"] = "test"
scored = F.add_scores(df, cal.predict_proba(df[F.CAT + F.NUM])[:, 1])
scored["recommended_action"] = scored.apply(F.recommended_action, axis=1)
tier = scored.groupby("risk_tier").agg(customers=("customerID", "size"),
                                       mean_prob=("churn_probability", "mean"),
                                       observed_churn=("target", "mean"),
                                       monthly_revenue=("MonthlyCharges", "sum"),
                                       revenue_at_risk=("revenue_at_risk_12m", "sum")).round(3)
tier
""")

code("""
(scored["risk_tier"].value_counts(normalize=True) * 100).round(2).to_dict()
""")

code("""
scored[scored["risk_tier"] == "High"].sort_values("churn_probability", ascending=False)[
    ["customerID", "tenure", "Contract", "PaymentMethod", "TechSupport", "MonthlyCharges",
     "churn_probability", "recommended_action"]].head(8)
""")

# --------------------------------------------------------------------------- 11. second source
md("""
## 11. Second source, and what drift can honestly be measured

The 11.1.3 release ships six modules. Reconciling it against the primary feed field by
field is what tells you whether the second file is new information or the same
information recoded.
""")

code("""
ibm = external.load_ibm_modules(Path("."))
{k: v.shape for k, v in ibm.items()}
""")

code("""
recon = external.reconcile(raw, ibm)
recon.to_string(index=False)
""")

md("""
The reconciliation is the honest part: `Dependents` disagrees on 551 rows (a genuine
cross-source conflict), the Q3 services module carries 265 contracts that the primary feed
codes as month-to-month, and it has no `Electronic check` category at all. Same 7,043
customer IDs, same churn labels.

Because the labels match, any drift computed on this batch is **upstream re-coding drift,
not temporal drift**. That is stated rather than glossed, and it is why the held-out test
split is included as a same-period control.
""")

code("""
q3 = external.q3_to_telco_schema(ibm)
print("mapped shape", q3.shape, "| conforms to the raw schema:", dq.validate_schema(q3, dq.telco_raw_schema())[0])
q3_engineered = F.engineer(q3)
q3_scored = F.add_scores(q3_engineered, cal.predict_proba(q3_engineered[F.CAT + F.NUM])[:, 1])
q3_scored["recommended_action"] = q3_scored.apply(F.recommended_action, axis=1)
heldout = scored[scored["split"] == "test"]
""")

code("""
ref = scored
notes = {"ibm_q3_rerelease": "IBM 11.1.3 Q3 re-release mapped to the 21-column schema. Same customers and "
                             "same labels as the reference; categories re-coded. Upstream re-coding drift, "
                             "NOT temporal drift.",
         "held_out_test_split": "Stratified test split, never used for training or calibration. Same period "
                                "as the reference by construction, so a near-zero PSI is a control."}
drift = monitor.build_drift_report(ref, {"ibm_q3_rerelease": q3_scored, "held_out_test_split": heldout},
                                   F.NUM, F.CAT, notes=notes)
drift[drift["measurement"].isin(["score_psi", "feature_psi_categorical:PaymentMethod",
                                 "category_change:PaymentMethod", "realised_roc_auc", "batch_churn_rate"])] \\
    [["batch", "measurement", "value", "verdict"]].to_string(index=False)
""")

md("""
A score-only PSI of ~0.02 would have said "nothing changed". Measuring the inputs too
attributes the change: `PaymentMethod` has a value-share PSI above 3, which is the
`Electronic check` category being removed upstream. PSI on a categorical cannot use the
quantile-binned formula - a 0/1 indicator has no deciles to bin - which is why
`monitor.psi_discrete` exists and `monitor.psi` is documented as numeric-only.
""")

code("""
bench = json.loads((OUT / "enrichment_benchmark_vs_ibm_score.json").read_text(encoding="utf-8"))
bench
""")

md("""
The second release also ships IBM's own `Churn Score` (5-96) and a `CLTV` field. On the
1,761 held-out rows IBM's score reaches a higher ROC-AUC than anything fitted here, which
is reported in full: the score is computed from inputs that are not in the published
schema, and it encodes the label, so it is excluded as a feature and used only as an
external yardstick. `CLTV` correlates 0.10 with monthly charges x 12, so it is a different
quantity, not a better one - both sizings stay on disk.
""")

code("""
enr = external.enrichment_frame(scored, ibm)
print("enrichment join:", enr.shape, "| State coverage", round(float(enr["State"].notna().mean()), 4))
enr[["customerID", "State", "Age", "Satisfaction Score", "CLTV", "Churn Category", "Churn Reason"]] \\
    [enr["target"] == 1].head(6)
""")

# --------------------------------------------------------------------------- 12. sql / api / tests
md("""
## 12. The same numbers out of SQL, HTTP and pytest

`sql/01_ingest.sql`, `02_dq_checks.sql` and `03_reporting.sql` re-implement ingestion, the
DQ card and the feature contract in DuckDB. `src/run_sql.py` executes them and diffs the
result against pandas. Reading the committed warehouse below is a live check that the two
engines still agree.
""")

code("""
import duckdb
con = duckdb.connect("warehouse/churn.duckdb", read_only=True)
con.execute("SELECT risk_tier, COUNT(*) AS n FROM core.scored GROUP BY 1 ORDER BY 2 DESC").df()
""")

code("""
con.execute('SELECT COUNT(*) AS blank_total_charges_landed_as_null FROM core.customer '
            'WHERE "TotalCharges" IS NULL').df()
""")

code("""
con.execute('SELECT "check", passed, detail FROM core.dq_scorecard WHERE NOT passed').df()
con.close()
""")

code("""
api = subprocess.run([sys.executable, "-m", "pytest", "tests/test_api.py", "-q"],
                     capture_output=True, text=True, cwd=str(Path(".").resolve()))
print(api.stdout.strip().splitlines()[-1])
""")

md("""
The endpoint itself is exercised live over HTTP - server started with
`python -m uvicorn src.api:app` and driven with curl - in `docs/api_smoke_test.md`. That
document exists because the first version of the endpoint returned 500 on every request
(`KeyError: 'tenure_band'`): the scoring helper coerced the derived categorical instead of
only the 15 the caller sends. `features.CAT_RAW` is the fix, and `tests/test_api.py` pins
that an API response and a committed batch row agree to four decimals.
""")

# --------------------------------------------------------------------------- 13. parity
md("""
## 13. Does this notebook agree with the pipeline?

Not a claim - a comparison against the committed artefacts, which `src/run_pipeline.py`
produced from the same code these cells call.
""")

code("""
committed = pd.read_csv(OUT / "model_comparison.csv").set_index("model")
mine = comp.set_index("model")
parity = pd.DataFrame({"notebook_roc_auc": mine["roc_auc"], "pipeline_roc_auc": committed["roc_auc"],
                       "notebook_pr_auc": mine["pr_auc"], "pipeline_pr_auc": committed["pr_auc"]})
parity["abs_diff"] = (parity["notebook_roc_auc"] - parity["pipeline_roc_auc"]).abs()
parity["abs_diff"] = parity["abs_diff"].combine(
    (parity["notebook_pr_auc"] - parity["pipeline_pr_auc"]).abs(), np.maximum)
parity
""")

code("""
worst = float(parity["abs_diff"].max())
ev = json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))
print("model zoo parity: max |notebook - pipeline| =", round(worst, 4))
assert worst < 0.005, "notebook and pipeline disagree - one of them is stale"
assert abs(brier_score_loss(y_test, p_cal) - ev["brier_calibrated"]) < 1e-4
assert abs(roc_auc_score(y_test, p_cal) - ev["roc_auc_calibrated"]) < 1e-4
assert round(float(best_thr_row["profit"]), 2) == ev["profit_at_threshold_scenario"], "profit curve disagrees"
assert round(float(best_thr_row["threshold"]), 2) == round(ev["profit_max_threshold"], 2)
assert sorted(scored["risk_tier"].value_counts().to_dict().items()) == \\
       sorted(pd.read_csv(OUT / "scored_customers.csv")["risk_tier"].value_counts().to_dict().items())
print("NOTEBOOK MATCHES PIPELINE - calibration, discrimination, profit threshold and tier counts all reproduce")
""")

# --------------------------------------------------------------------------- 14. limitations
md("""
## 14. Limitations, stated before anyone asks

- **Snapshot, not a panel.** No signup date, no usage series, no ticket log. Time-based
  splits and true cohorts are impossible here; the tenure-observed convention is documented
  in the survival section and KM by contract is descriptive only.
- **Bill shock and contract-remaining are proxies**, built from the 21 published columns.
  They are labelled as proxies in the data dictionary, not passed off as measured fields.
- **Scenario profit is a scenario.** 30% save rate, \$40 per contact, CLV = monthly x 12.
  No uplift claim survives without an A/B test.
- **The second source is the same book of business re-coded.** Drift measured on it is
  upstream re-coding, and the held-out split is published alongside it as a control so the
  reader can see the difference between "no drift" and "no new period".
- **IBM's supplied Churn Score beats this build on the held-out rows.** Reported with the
  caveat that it uses inputs that are not published and encodes the label.
- **Fairness coverage is partial.** `SeniorCitizen`, `gender` and `Dependents` are the only
  near-protected attributes in the feed; there is no race, ethnicity or national-origin
  field, so no claim about those groups is made.
- **Validation is Pandera**, deliberately: one framework, 21 columns covered by schema plus
  business checks, mirrored in SQL. No Great Expectations dependency exists in this repo.

## Reproduce

```bash
python src/run_pipeline.py                                   # models, calibration, SHAP, fairness, drift
Rscript r/survival_analysis.R                                # R KM + Cox -> outputs/r/
python src/run_sql.py                                        # DuckDB build + pandas parity checks
python src/score_batch.py --ibm-q3                           # second-source batch scoring
python -m uvicorn src.api:app --port 8000                    # scoring API
streamlit run dashboard/app.py                               # dashboard
python -m pytest                                             # 138 tests
python src/make_notebook.py --execute                        # this notebook
```
""")


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (churn-platform)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb.cells = [nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
                for kind, text in cells]
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="run the notebook with nbconvert after writing it")
    args = ap.parse_args()

    nb = build()
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH.name} with {len(nb.cells)} cells ({sum(1 for k, _ in cells if k == 'code')} code)")

    if args.execute:
        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=600", "--ExecutePreprocessor.kernel_name=python3",
               str(NB_PATH)]
        print(" ".join(cmd))
        return subprocess.call(cmd, cwd=str(BASE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
