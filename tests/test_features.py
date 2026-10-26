"""Feature engineering, on a 6-row hand-computed frame plus the real snapshot.

Every expected value below is arithmetic anyone can check by hand, so a change to
engineer() that alters a definition fails here rather than altering a scorecard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features as F

COLS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]


def make(rows: list[dict]) -> pd.DataFrame:
    base = {c: "No" for c in COLS}
    base.update({"customerID": "0000-AAAAA", "gender": "Female", "SeniorCitizen": "0",
                 "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
                 "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
                 "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                 "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
                 "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                 "MonthlyCharges": "50.0", "TotalCharges": "100.0", "Churn": "Yes",
                 "tenure": "2"})
    return pd.DataFrame([{**base, **r} for r in rows])[COLS]


def test_service_counts_yes_across_the_nine_service_columns():
    df = F.engineer(make([{"PhoneService": "Yes", "MultipleLines": "Yes", "TechSupport": "Yes",
                           "OnlineSecurity": "Yes", "InternetService": "DSL"}]))
    # InternetService='DSL' is not 'Yes', so it does not count - deliberate, matches SQL.
    assert int(df["service_count"][0]) == 4


def test_tenure_zero_uses_monthly_as_avg_spend_not_a_division_by_zero():
    df = F.engineer(make([{"tenure": "0", "TotalCharges": " ", "MonthlyCharges": "70.0"}]))
    assert df["TotalCharges"][0] == 0.0
    assert df["avg_monthly_spend"][0] == 70.0
    assert df["bill_shock_pct"][0] == 0.0
    assert df["is_new"][0] == 1


def test_bill_shock_uses_the_customers_own_history():
    # avg spend 100/2 = 50, current 66 -> +32% -> flagged.
    df = F.engineer(make([{"tenure": "2", "MonthlyCharges": "66.0", "TotalCharges": "100.0"}]))
    assert df["avg_monthly_spend"][0] == 50.0
    assert df["bill_shock_pct"][0] == pytest.approx(0.32)
    assert df["bill_shock_flag"][0] == 1


def test_bill_shock_threshold_is_strictly_above_twenty_percent():
    at = F.engineer(make([{"tenure": "2", "MonthlyCharges": "60.0", "TotalCharges": "100.0"}]))
    assert at["bill_shock_pct"][0] == pytest.approx(0.20)
    assert int(at["bill_shock_flag"][0]) == 0


def test_contract_risk_and_remaining_are_ordinal_and_wrapped():
    df = F.engineer(make([
        {"Contract": "Month-to-month", "tenure": "30", "TotalCharges": "1500"},
        {"Contract": "One year", "tenure": "14", "TotalCharges": "700", "customerID": "0001-BBBBB"},
        {"Contract": "Two year", "tenure": "25", "TotalCharges": "1250", "customerID": "0002-CCCCC"},
    ]))
    assert df["contract_risk"].tolist() == [2, 1, 0]
    # One year at tenure 14 -> 12 - (14 % 12) = 10; Two year at 25 -> 24 - (25 % 24) = 23.
    assert df["contract_remaining_proxy"].tolist() == [0, 10, 23]
    assert df["electronic_check_flag"].tolist() == [1, 1, 1]


def test_flags_read_absence_not_the_no_internet_sentinel():
    df = F.engineer(make([{"InternetService": "No", "TechSupport": "No internet service",
                           "OnlineSecurity": "No internet service"}]))
    assert int(df["no_support_flag"][0]) == 0
    assert int(df["no_security_flag"][0]) == 0


def test_tenure_bands_match_the_declared_bins():
    df = F.engineer(make([
        {"tenure": "0"}, {"tenure": "12"}, {"tenure": "13"}, {"tenure": "24"},
        {"tenure": "25"}, {"tenure": "48"}, {"tenure": "49"}, {"tenure": "72"},
    ]))
    assert df["tenure_band"].astype(str).tolist() == [
        "0-12", "0-12", "13-24", "13-24", "25-48", "25-48", "49-72", "49-72"]


def test_log_total_charges_is_log1p():
    df = F.engineer(make([{"TotalCharges": "1000.0", "tenure": "20"}]))
    assert df["log_total_charges"][0] == pytest.approx(np.log1p(1000.0))


def test_charges_per_service_clips_the_denominator():
    df = F.engineer(make([{"MonthlyCharges": "30.0", "PhoneService": "No"}]))
    assert int(df["service_count"][0]) == 0
    assert df["charges_per_service"][0] == pytest.approx(30.0)  # /max(0,1), not /0


def test_target_is_derived_from_the_label():
    df = F.engineer(make([{"Churn": "No"}, {"Churn": "Yes", "customerID": "0001-BBBBB"}]))
    assert df["target"].tolist() == [0, 1]


def test_model_contract_rejects_a_missing_feature():
    df = F.engineer(make([{}]))
    F.assert_model_contract(df)  # passes
    with pytest.raises(KeyError, match="bill_shock_flag"):
        F.assert_model_contract(df.drop(columns=["bill_shock_flag"]))


@pytest.mark.parametrize("prob,tier", [(0.60, "High"), (0.5999, "Medium"), (0.35, "Medium"),
                                       (0.3499, "Low"), (1.0, "High"), (0.0, "Low")])
def test_risk_tier_cutoffs_are_inclusive_at_the_top(prob, tier):
    assert F.risk_tier(prob) == tier


def test_recommended_action_precedence():
    base = F.engineer(make([{}])).iloc[0].to_dict()
    high = {**base, "risk_tier": "High", "Contract": "Month-to-month", "electronic_check_flag": 1,
            "no_support_flag": 1, "bill_shock_flag": 1}
    assert F.recommended_action(high) == "Proactive support call + contract upgrade offer"
    assert F.recommended_action({**high, "risk_tier": "Low"}) == "Monitor / standard care"
    assert F.recommended_action({**high, "electronic_check_flag": 0}) == \
        "Proactive support call + tech-support bundle"
    assert F.recommended_action({**high, "electronic_check_flag": 0, "no_support_flag": 0}) == \
        "Discount / bill-review offer"
    assert F.recommended_action({**high, "electronic_check_flag": 0, "no_support_flag": 0,
                                 "bill_shock_flag": 0, "tenure": 6}) == \
        "Product education + onboarding check-in"
    assert F.recommended_action({**high, "electronic_check_flag": 0, "no_support_flag": 0,
                                 "bill_shock_flag": 0, "tenure": 40}) == "Loyalty reward offer"


def test_add_scores_fills_revenue_only_for_high_tier(engineered):
    out = F.add_scores(engineered.head(50), np.linspace(0.0, 1.0, 50))
    assert out["churn_probability"].between(0, 1).all()
    assert (out.loc[out["risk_tier"] != "High", "revenue_at_risk_12m"] == 0).all()
    hi = out[out["risk_tier"] == "High"]
    if len(hi):
        assert hi["revenue_at_risk_12m"].iloc[0] == pytest.approx(
            hi["MonthlyCharges"].iloc[0] * F.CLV_MONTHS)


def test_feature_contract_has_no_leaky_columns():
    """The second source fields that encode the answer must never reach the model."""
    banned = {"Churn Score", "CLTV", "Churn Reason", "Churn Category", "Satisfaction Score",
              "Customer Status", "Churn", "target", "Churn Value"}
    assert not (banned & set(F.CAT + F.NUM))
    assert len(F.CAT) == 16 and len(F.NUM) == 16


def test_preprocessor_column_count_matches_the_contract(engineered):
    pre = F.build_preprocessor()
    Xt = pre.fit_transform(engineered[F.CAT + F.NUM])
    n_cat_levels = sum(len(c) for c in pre.transformers_[0][1].categories_)
    assert Xt.shape == (len(engineered), n_cat_levels + len(F.NUM))


def test_engineer_is_idempotent_on_the_real_data(raw_df, engineered):
    """Re-running engineer() on its own output must not shift any feature."""
    again = F.engineer(engineered.astype(raw_df.dtypes.to_dict())
                       .assign(TotalCharges=lambda d: d["TotalCharges"].astype(str)))
    for c in F.NUM:
        assert np.allclose(pd.to_numeric(again[c], errors="coerce"),
                           pd.to_numeric(engineered[c], errors="coerce"), atol=1e-9), c
