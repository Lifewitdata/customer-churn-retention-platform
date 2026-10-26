"""Second-source tests (items 12/13).

The IBM 11.1.3 release is the second feed. The tests here exist to keep the two claims
about it honest: it conforms to the same 21-column contract as the primary feed (so the
drift batch is a real batch, not a special case), and it is a re-coded re-release of the
same book of business with the same labels (so drift measured on it is upstream
re-coding drift, never temporal drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dq
import external
import features as F

BASE = Path(external.__file__).resolve().parents[1]
RAW_CSV = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def ibm():
    return external.load_ibm_modules(BASE)


@pytest.fixture(scope="module")
def primary():
    return F.load_raw(RAW_CSV)


@pytest.fixture(scope="module")
def q3(ibm):
    return external.q3_to_telco_schema(ibm)


def test_all_six_modules_load_and_are_customer_level(ibm):
    assert set(ibm) == set(external.MODULES)
    assert all(len(df) > 0 for df in ibm.values())
    for key in ("main", "services", "demographics", "status"):
        assert len(ibm[key]) == 7043, key
    # location/population are zip-level, so they join through Zip Code, not customerID
    assert "Zip Code" in ibm["population"].columns


def test_the_q3_batch_conforms_to_the_same_raw_schema_as_the_primary_feed(q3):
    """The drift batch is not a special case: the identical Pandera schema that guards the
    primary CSV must accept the mapped second source."""
    assert list(q3.columns) == dq.EXPECTED_COLUMNS
    assert len(q3) == 7043
    ok, detail = dq.validate_schema(q3, dq.telco_raw_schema())
    assert ok, detail


def test_labels_are_identical_across_the_two_feeds(primary, q3):
    """Why the drift report cannot claim temporal drift: same customers, same outcomes."""
    merged = primary.merge(q3[["customerID", "Churn"]], on="customerID", suffixes=("_p", "_q"))
    assert len(merged) == 7043
    assert (merged["Churn_p"] == merged["Churn_q"]).all()


def test_q3_recodes_payment_and_contract_domains(q3, primary):
    assert "Electronic check" not in set(q3["PaymentMethod"])
    assert "Electronic check" in set(primary["PaymentMethod"])
    assert set(q3["PaymentMethod"]) <= set(dq.ALLOWED["PaymentMethod"])
    assert set(q3["Contract"]) <= set(dq.ALLOWED["Contract"])
    # the re-coded rows are the ones the drift report should attribute
    moved = primary.merge(q3[["customerID", "Contract"]], on="customerID", suffixes=("_p", "_q"))
    changed = moved[moved["Contract_p"] != moved["Contract_q"]]
    assert (changed["Contract_p"] == "Month-to-month").all()
    assert len(changed) == 265


def test_unmapped_category_raises_instead_of_scoring_silently(ibm, monkeypatch):
    """If a future release adds a payment method the map does not cover, the mapping must
    fail loudly; a NaN category would encode to an all-zero one-hot row."""
    monkeypatch.setattr(external, "PAYMENT_MAP", {"Bank Withdrawal": "Bank transfer (automatic)"})
    with pytest.raises(ValueError, match="unmapped category"):
        external.q3_to_telco_schema(ibm)


def test_reconciliation_table_is_bounded_and_surfaces_the_conflicts(ibm, primary):
    rec = external.reconcile(primary, ibm)
    assert rec["value"].between(0, 1).all()
    by = rec.set_index(["comparison", "metric"])["value"]
    assert by[("customerID join coverage vs main", "matched rows / primary rows")] == 1.0
    # 551 rows where the two feeds disagree on Dependents is a genuine cross-source
    # conflict, and it must stay visible rather than being averaged away
    assert by[("Dependents vs demographics Dependents", "exact match rate")] < 0.95
    assert by[("PaymentMethod vs services Payment Method",
               "match rate after documented map")] < 0.55


def test_committed_reconciliation_matches_a_fresh_run(ibm, primary):
    """The reconciliation card is a committed artefact, so it has to regenerate bit-for-bit
    from the two feeds. Nothing here is allowed to be a screenshot of an old run."""
    committed = pd.read_csv(BASE / "outputs" / "source_reconciliation.csv").fillna("")
    fresh = external.reconcile(primary, ibm).fillna("")
    assert list(committed.columns) == list(fresh.columns)
    assert committed[["comparison", "metric", "detail"]].astype(str).values.tolist() ==         fresh[["comparison", "metric", "detail"]].astype(str).values.tolist()
    np.testing.assert_allclose(fresh["value"].astype(float).values,
                               committed["value"].astype(float).values)


def test_enrichment_joins_without_losing_rows_and_keeps_labels_only_fields_out(scored, ibm):
    enr = external.enrichment_frame(scored, ibm)
    assert len(enr) == len(scored)
    assert enr["State"].notna().mean() > 0.99
    for col in ("Age", "Number of Dependents", "State", "Satisfaction Score", "CLTV",
                "Churn Category", "Churn Reason", "Customer Status"):
        assert col in enr.columns, col
    # Churn Score reaches the report through the benchmark block, not the enrichment join
    assert "Churn Score" in ibm["status"].columns and "Churn Score" not in enr.columns
    # leakage guard: none of the post-outcome or second-source fields is a model input
    assert not (set(external.ENRICHMENT_ONLY) & set(F.CAT + F.NUM))
    assert "Churn Score" in external.ENRICHMENT_ONLY


def test_churn_reason_exists_only_for_churners(scored, ibm):
    enr = external.enrichment_frame(scored, ibm)
    churners = enr[enr["target"] == 1]
    assert churners["Churn Reason"].notna().all()
    assert enr[enr["target"] == 0]["Churn Reason"].isna().all()


def test_cltv_is_not_a_stand_in_for_the_spend_proxy(scored, ibm):
    """The uncomfortable number the build reports instead of hiding: IBM's CLTV correlates
    0.10 with MonthlyCharges x 12 and 0.34 with MonthlyCharges x tenure, so it is a
    different quantity, not a better one. Both sizings stay on disk and neither is a
    feature."""
    enr = external.enrichment_frame(scored, ibm)
    r12 = float(enr["CLTV"].corr(enr["MonthlyCharges"] * 12))
    assert r12 < 0.15
    assert r12 == pytest.approx(
        json.loads((BASE / "outputs" / "enrichment_cltv_vs_proxy.json").read_text(encoding="utf-8"))
        ["pearson_r_monthly_x_12_vs_cltv"], abs=1e-4)
    assert float(enr["CLTV"].corr(enr["MonthlyCharges"] * enr["tenure"])) < 0.5
