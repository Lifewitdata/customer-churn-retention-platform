"""Scoring tests: the calibrated artefact, the tier/action rules, and batch parity.

The load-bearing claim in this repo is that every surface (pipeline CSV, batch job,
API, dashboard, SQL) reports the *same* churn_probability for the same customer,
computed by the model the pipeline actually saved. test_full_snapshot_parity is the
test that protects that; everything else pins the rule layer around it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

import features as F
import score_batch

BASE = score_batch.BASE
EVAL = json.loads((BASE / "outputs" / "evaluation.json").read_text(encoding="utf-8"))


def test_deployed_artefact_is_the_calibrated_model(calibrated_model):
    assert isinstance(calibrated_model, CalibratedClassifierCV)
    assert calibrated_model.method == "isotonic"
    assert calibrated_model.classes_.tolist() == [0, 1]


def test_calibrated_model_beats_the_uncalibrated_one_on_brier(calibrated_model):
    """Item 5: calibration must be used for scoring, not just fitted and left on disk.
    Isotonic reshapes probabilities without relearning the ranking, so AUC moves only by
    the amount its step function ties together - a few tenths of a point - while Brier
    improves by much more."""
    best = score_batch.joblib.load(BASE / "outputs" / "model_best.pkl")
    scored = pd.read_csv(BASE / "outputs" / "scored_customers.csv")
    test = scored[scored["split"] == "test"]
    X = test[F.CAT + F.NUM]
    p_cal = calibrated_model.predict_proba(X)[:, 1]
    p_raw = best.predict_proba(X)[:, 1]
    y = test["target"].values
    assert brier_score_loss(y, p_cal) < brier_score_loss(y, p_raw)
    assert abs(roc_auc_score(y, p_cal) - roc_auc_score(y, p_raw)) < 0.005
    assert abs(EVAL["brier_uncalibrated"] - brier_score_loss(y, p_raw)) < 1e-4


def test_committed_evaluation_numbers_reproduce(scored):
    """outputs/evaluation.json is not allowed to drift from the CSV it was written with."""
    test = scored[scored["split"] == "test"]
    assert len(test) == EVAL["test_rows"]
    assert pytest.approx(brier_score_loss(test["target"], test["churn_probability"]), abs=1e-4) == EVAL["brier_calibrated"]
    assert pytest.approx(roc_auc_score(test["target"], test["churn_probability"]), abs=1e-4) == EVAL["roc_auc_calibrated"]
    assert pytest.approx(float((test["churn_probability"] >= F.HIGH_RISK_CUTOFF).mean()), abs=1e-4) == EVAL["high_risk_share_test"]


def test_full_snapshot_parity_between_saved_model_and_committed_csvs(calibrated_model, scored):
    """Re-scoring all 7,043 rows with the deployed artefact must reproduce the committed
    column exactly. This is the bug that would otherwise ship silently: a dashboard,
    batch or API reading a model that is not the one the pipeline saved."""
    recomputed = calibrated_model.predict_proba(scored[F.CAT + F.NUM])[:, 1]
    np.testing.assert_allclose(recomputed, scored["churn_probability"].values, atol=1e-9)


@pytest.mark.parametrize("p,expected", [
    (0.60, "High"), (0.5999, "Medium"), (0.35, "Medium"), (0.3499, "Low"),
    (1.0, "High"), (0.0, "Low")])
def test_risk_tier_cutoffs_are_inclusive_at_the_bottom(p, expected):
    assert F.risk_tier(p) == expected


def test_add_scores_populates_only_the_high_tier_with_value_at_risk():
    frame = pd.DataFrame({"MonthlyCharges": [100.0, 50.0, 80.0]})
    out = F.add_scores(frame, [0.61, 0.20, 0.90])
    assert out["risk_tier"].tolist() == ["High", "Low", "High"]
    assert out["revenue_at_risk_12m"].tolist() == [100 * F.CLV_MONTHS, 0.0, 80 * F.CLV_MONTHS]


def build_row(**over):
    """A minimal but valid Telco-format row; each test overrides only what its branch needs."""
    row = dict(Contract="Month-to-month", electronic_check_flag=0, no_support_flag=0,
               bill_shock_flag=0, tenure=48, risk_tier="High")
    row.update(over)
    return pd.Series(row)


@pytest.mark.parametrize("over,expected", [
    ({}, "Loyalty reward offer"),
    (dict(electronic_check_flag=1), "Proactive support call + contract upgrade offer"),
    (dict(Contract="One year", electronic_check_flag=1, no_support_flag=1), "Proactive support call + tech-support bundle"),
    (dict(Contract="One year", electronic_check_flag=1), "Loyalty reward offer"),
    (dict(no_support_flag=1, bill_shock_flag=1), "Proactive support call + tech-support bundle"),
    (dict(bill_shock_flag=1), "Discount / bill-review offer"),
    (dict(tenure=6), "Product education + onboarding check-in"),
    (dict(bill_shock_flag=1, tenure=6), "Discount / bill-review offer"),
    (dict(Contract="Two year", electronic_check_flag=1, tenure=6), "Product education + onboarding check-in"),
])
def test_recommended_action_precedence(over, expected):
    """Order matters and is fixed: the contract+payment branch first, then an unmet
    support need, then bill shock, then new-customer onboarding, else loyalty. A
    month-to-month payer on electronic check with no support gap therefore gets the call,
    not the discount."""
    assert F.recommended_action(build_row(**over)) == expected


def test_non_high_tier_always_gets_standard_care():
    for tier in ("Medium", "Low"):
        assert F.recommended_action(build_row(risk_tier=tier, electronic_check_flag=1,
                                              no_support_flag=1, bill_shock_flag=1)) == "Monitor / standard care"


def test_scenario_profit_is_the_documented_arithmetic():
    """10 churners at $50/mo saved at 30% over 12 months, minus $40 per contacted row."""
    sub = pd.DataFrame({"target": [1] * 10 + [0] * 10, "MonthlyCharges": [50.0] * 20})
    expected = 10 * 50.0 * F.CLV_MONTHS * F.SCENARIO_SAVE_RATE - 20 * F.CAMPAIGN_COST_PER_CUSTOMER
    assert F.scenario_profit(sub) == pytest.approx(expected)
    assert expected == 1800.0 - 800.0


def test_scenario_profit_goes_negative_when_contact_cost_exceeds_retained_value():
    sub = pd.DataFrame({"target": [1], "MonthlyCharges": [10.0]})
    profit = F.scenario_profit(sub)
    assert profit == pytest.approx(10.0 * F.CLV_MONTHS * F.SCENARIO_SAVE_RATE - F.CAMPAIGN_COST_PER_CUSTOMER)
    assert profit < 0


def test_score_batch_scores_a_raw_frame_and_returns_the_contract_columns(raw_df, calibrated_model):
    out = score_batch.score(F.load_raw(score_batch.BASE / "data" / "raw" /
                                      "WA_Fn-UseC_-Telco-Customer-Churn.csv"), calibrated_model)
    for col in ("churn_probability", "risk_tier", "revenue_at_risk_12m", "target"):
        assert col in out.columns
    assert len(out) == 7043
    assert out["churn_probability"].between(0, 1).all()
    assert (out["risk_tier"] == out["churn_probability"].apply(F.risk_tier)).all()


def test_score_is_deterministic(raw_df, calibrated_model):
    a = score_batch.score(F.load_raw(score_batch.BASE / "data" / "raw" /
                                     "WA_Fn-UseC_-Telco-Customer-Churn.csv"), calibrated_model)
    b = score_batch.score(F.load_raw(score_batch.BASE / "data" / "raw" /
                                     "WA_Fn-UseC_-Telco-Customer-Churn.csv"), calibrated_model)
    np.testing.assert_array_equal(a["churn_probability"].values, b["churn_probability"].values)


def test_model_comparison_contains_the_pytorch_row():
    """Item 1: the feedforward network has to appear as a compared row, not prose."""
    cmp = pd.read_csv(BASE / "outputs" / "model_comparison.csv")
    assert "Feedforward NN (PyTorch)" in set(cmp["model"])
    row = cmp[cmp["model"] == "Feedforward NN (PyTorch)"].iloc[0]
    assert 0.5 < row["roc_auc"] < 1.0 and 0.0 < row["pr_auc"] < 1.0
    assert cmp["roc_auc"].notna().all() and (cmp["roc_auc"] < 1.0).all()
    # the torch net must not be a strawman: it should beat the sklearn MLP it replaces
    assert row["roc_auc"] > float(cmp.loc[cmp["model"] == "MLP (sklearn feedforward)", "roc_auc"].iloc[0])


def test_saved_torch_model_loads_and_scores_the_contract_frame(engineered, calibrated_model):
    """The PyTorch artefact is pickled through the sklearn wrapper, so it must load and
    accept exactly the 32 columns features.CAT + features.NUM produce."""
    nn_model = score_batch.joblib.load(BASE / "outputs" / "model_Feedforward_NN.pkl")
    X = engineered[F.CAT + F.NUM].head(50)
    proba = nn_model.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.isclose(proba.sum(axis=1), 1.0).all()
    assert set(nn_model.feature_names_in_) == set(F.CAT + F.NUM)
    # ranking agreement with the deployed model: same problem, different family
    from scipy.stats import spearmanr
    assert spearmanr(proba[:, 1], calibrated_model.predict_proba(X)[:, 1]).statistic > 0.85
