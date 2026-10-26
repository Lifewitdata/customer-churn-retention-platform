"""DQ layer: the real snapshot must pass, and every check must fail when broken.

The second half matters more than the first. A scorecard that is 35/35 on clean data
proves nothing about the checks; each one here is inverted deliberately and must catch
it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dq
import features as F


def failed(card: pd.DataFrame) -> set[str]:
    return set(card.loc[~card["passed"], "check"])


def test_real_snapshot_is_clean(raw_df):
    card = dq.run_dq_checks(raw_df)
    assert dq.score_scorecard(card) == 100.0, card[~card["passed"]].to_string(index=False)


def test_scorecard_covers_every_check(raw_df):
    card = dq.run_dq_checks(raw_df)
    assert len(card) == 35
    assert set(card["layer"]) == {"business", "schema"}
    assert card["check"].is_unique
    assert card["detail"].notna().all()


def test_expected_columns_are_the_21_documented_ones():
    assert len(dq.EXPECTED_COLUMNS) == 21
    assert F.CAT + F.NUM != dq.EXPECTED_COLUMNS  # features are derived, not raw


@pytest.mark.parametrize("column,poison", [
    ("PaymentMethod", "PayPal"),
    ("Contract", "Monthly"),
    ("InternetService", "Cable"),
    ("gender", "Other"),
    ("Churn", "maybe"),
])
def test_illegal_category_is_caught(corrupt, column, poison):
    corrupt.loc[0, column] = poison
    assert f"domain_{column}" in failed(dq.run_dq_checks(corrupt))


def test_blank_totalcharges_on_a_tenured_customer_is_caught(corrupt):
    # Legitimate blanks exist on tenure-0 rows; moving one is the bug this detects.
    tenured = corrupt.index[corrupt["tenure"].astype(int) > 0][0]
    corrupt.loc[tenured, "TotalCharges"] = " "
    card = dq.run_dq_checks(corrupt)
    assert "blank_TotalCharges_only_tenure_0" in failed(card)


def test_duplicate_customer_id_is_caught(corrupt):
    corrupt.loc[corrupt.index[1], "customerID"] = corrupt.loc[corrupt.index[0], "customerID"]
    assert "duplicate_customerID_0" in failed(dq.run_dq_checks(corrupt))


def test_tenure_out_of_window_is_caught(corrupt):
    corrupt.loc[5, "tenure"] = "80"
    assert "tenure_range_0_72" in failed(dq.run_dq_checks(corrupt))


def test_zero_monthly_charge_is_caught(corrupt):
    corrupt.loc[7, "MonthlyCharges"] = "0"
    assert "MonthlyCharges_positive" in failed(dq.run_dq_checks(corrupt))


def test_negative_total_charge_is_caught(corrupt):
    corrupt.loc[9, "TotalCharges"] = "-50"
    assert "TotalCharges_never_negative" in failed(dq.run_dq_checks(corrupt))


def test_phone_addon_inconsistency_is_caught(corrupt):
    corrupt.loc[corrupt["PhoneService"] == "No", "MultipleLines"] = "Yes"
    assert "phone_addon_consistency" in failed(dq.run_dq_checks(corrupt))


def test_internet_addon_inconsistency_is_caught(corrupt):
    no_net = corrupt["InternetService"] == "No"
    corrupt.loc[no_net, "OnlineSecurity"] = "Yes"
    assert "internet_addon_consistency" in failed(dq.run_dq_checks(corrupt))


def test_totalcharges_far_off_monthly_x_tenure_is_caught(corrupt):
    corrupt["TotalCharges"] = (pd.to_numeric(corrupt["MonthlyCharges"])
                               * pd.to_numeric(corrupt["tenure"]) * 2).astype(str)
    assert "TotalCharges_within_25pct_of_monthly_x_tenure" in failed(dq.run_dq_checks(corrupt))


def test_malformed_customer_id_breaks_the_raw_schema(corrupt):
    corrupt.loc[3, "customerID"] = "abc-123"
    assert "pandera_schema_telco_raw" in failed(dq.run_dq_checks(corrupt))


def test_clean_schema_rejects_an_out_of_range_tenure(raw_df):
    clean = F.coerce_numeric(raw_df)
    clean.loc[0, "tenure"] = 999
    ok, detail = dq.validate_schema(clean, dq.telco_clean_schema())
    assert not ok and "tenure" in detail


def test_clean_schema_passes_on_the_real_data(raw_df):
    ok, detail = dq.validate_schema(F.coerce_numeric(raw_df), dq.telco_clean_schema())
    assert ok, detail


def test_blank_totalcharges_stays_visible_only_with_dtype_str(raw_df):
    """Blank TotalCharges is the only legitimate blank in the file, and the reason
    load_raw pins dtype=str, keep_default_na=False: the caller decides what a blank
    means, not the CSV parser."""
    from conftest import RAW_CSV
    blanks = (raw_df["TotalCharges"].astype(str).str.strip() == "").sum()
    assert blanks == 11
    naive = pd.read_csv(RAW_CSV)
    # pandas 3 leaves " " as a string, so the default read hands back a text column where
    # every downstream numeric use expects a float; pandas 2 turned the same cell into
    # NaN. Neither is a column you can aggregate, which is the point of load_raw.
    assert not pd.api.types.is_numeric_dtype(naive["TotalCharges"])
    assert F.coerce_numeric(raw_df)["TotalCharges"].isna().sum() == 11
    assert raw_df.loc[raw_df["TotalCharges"].astype(str).str.strip() == "", "tenure"].eq("0").all()
