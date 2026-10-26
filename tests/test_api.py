"""FastAPI endpoint tests (item 15).

These exist because the endpoint was shipped once on the strength of "it imports" and
POST /score actually returned 500 (KeyError: 'tenure_band'). Two things are asserted
here that a smoke test would miss:

1. Domain validation. OneHotEncoder runs with handle_unknown='ignore', so an
   out-of-domain category encodes to an all-zero row and scores as a confident,
   meaningless number. A typo must be a 422, never a business decision.
2. Parity with the batch. The probability the API returns for a customer must equal
   the churn_probability already committed in outputs/scored_customers.csv, or the
   endpoint is serving a different model than the pipeline and dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api as API
import features as F


@pytest.fixture(scope="module")
def client():
    return TestClient(API.app)


def payload(row: pd.Series) -> dict:
    """Project a Telco-format row (raw strings or the scored CSV) onto the fields the
    endpoint accepts. Churn/target/split are labels and pipeline output, never input."""
    p = {c: str(row[c]) for c in F.CAT_RAW}
    p.update(customerID=str(row["customerID"]), SeniorCitizen=int(row["SeniorCitizen"]),
             tenure=int(row["tenure"]), MonthlyCharges=float(row["MonthlyCharges"]),
             TotalCharges=float(row["TotalCharges"]))
    return p


def test_health_reports_the_deployed_artifact(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "model_calibrated.pkl"
    assert body["model_present"] is True
    assert body["n_features"] == len(F.CAT) + len(F.NUM) == 32
    assert (body["high_risk_cutoff"], body["medium_risk_cutoff"]) == (0.6, 0.35)


def test_score_returns_every_downstream_field(client, raw_df):
    row = raw_df.iloc[0]
    res = client.post("/score", json=payload(row))
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"customerID", "churn_probability", "risk_tier", "recommended_action",
                         "revenue_at_risk_12m", "model"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"High", "Medium", "Low"}
    assert body["recommended_action"]


def test_score_matches_the_committed_batch(client, scored):
    """Same customer through the API and through the pipeline must agree."""
    picked = scored.loc[scored["churn_probability"].between(0.45, 0.75)
                        & (scored["TotalCharges"] > 0)].head(25)
    assert len(picked) == 25, "need a spread of mid-risk customers to compare against"
    for _, row in picked.iterrows():
        body = client.post("/score", json=payload(row)).json()
        assert body["churn_probability"] == pytest.approx(row["churn_probability"], abs=1e-4), row["customerID"]
        assert body["risk_tier"] == row["risk_tier"]
        assert body["recommended_action"] == row["recommended_action"]


def test_revenue_at_risk_is_zero_below_the_high_cutoff(client, raw_df):
    low = raw_df[(raw_df["Contract"] == "Two year") & (raw_df["tenure"].astype(int) > 60)].iloc[0]
    body = client.post("/score", json=payload(low)).json()
    if body["risk_tier"] != "High":
        assert body["revenue_at_risk_12m"] == 0.0
    else:
        assert body["revenue_at_risk_12m"] == pytest.approx(
            round(float(low["MonthlyCharges"]) * F.CLV_MONTHS, 2))


@pytest.mark.parametrize("field,bad", [
    ("PaymentMethod", "Email check"),          # removed in the Q3 re-code; must not silently zero-encode
    ("Contract", "Monthly"),
    ("InternetService", "Fibre"),
    ("gender", "Other"),
    ("OnlineSecurity", "maybe"),
])
def test_out_of_domain_category_is_rejected(client, raw_df, field, bad):
    body = payload(raw_df.iloc[0])
    body[field] = bad
    assert client.post("/score", json=body).status_code == 422


@pytest.mark.parametrize("field,bad", [
    ("tenure", 120), ("tenure", -1), ("SeniorCitizen", 2),
    ("MonthlyCharges", 0), ("MonthlyCharges", -12.5), ("TotalCharges", -1),
])
def test_out_of_range_numeric_is_rejected(client, raw_df, field, bad):
    body = payload(raw_df.iloc[0])
    body[field] = bad
    assert client.post("/score", json=body).status_code == 422


def test_partial_payload_uses_the_documented_defaults(client):
    """Every field is optional-with-default by design (the docstring advertises
    "any subset"), so a one-field request must score, not 422. What the contract
    protects is the *value* domain, tested above."""
    partial = client.post("/score", json={"customerID": "1111-AAAA"})
    assert partial.status_code == 200
    assert np.isfinite(partial.json()["churn_probability"])
    # the default combination must itself be a valid customer (no illegal
    # "No internet service" vs "Yes" pairing) and identical to the sparse request
    explicit = client.post("/score", json=API.Customer().model_dump())
    assert explicit.status_code == 200
    assert explicit.json()["churn_probability"] == partial.json()["churn_probability"]


def test_blank_total_charges_matches_the_pipeline_that_filled_zero(client, scored, raw_df):
    """The 11 tenure-0 rows have a blank TotalCharges; engineer() fills 0 and the batch
    commits 0.0. The API takes a float, so a caller that sends 0 must land on the
    probability the batch recorded."""
    blank = raw_df[raw_df["TotalCharges"].str.strip() == ""]
    assert len(blank) == 11
    ids = set(blank["customerID"])
    for _, row in scored[scored["customerID"].isin(ids)].iterrows():
        assert int(row["tenure"]) == 0
        body = client.post("/score", json=payload(row)).json()
        assert np.isfinite(body["churn_probability"])
        assert body["churn_probability"] == pytest.approx(row["churn_probability"], abs=1e-4)


def test_non_numeric_total_charges_is_rejected(client, raw_df):
    body = payload(raw_df.iloc[0]) | {"TotalCharges": ""}
    assert client.post("/score", json=body).status_code == 422


def test_batch_scores_and_counts_tiers(client, raw_df):
    rows = raw_df.head(200)
    res = client.post("/score/batch", json=[payload(r) for _, r in rows.iterrows()])
    assert res.status_code == 200
    body = res.json()
    assert body["n"] == 200
    assert sum(body["counts_by_tier"].values()) == 200
    assert {t for t in body["counts_by_tier"]} <= {"High", "Medium", "Low"}
    assert len(body["customers"]) == 200
    for c in body["customers"][:20]:
        assert c["risk_tier"] == F.risk_tier(c["churn_probability"])
        assert c["recommended_action"]


def test_batch_matches_single_score(client, raw_df):
    rows = [payload(r) for _, r in raw_df.sample(40, random_state=7).iterrows()]
    batch = client.post("/score/batch", json=rows).json()["customers"]
    singles = [client.post("/score", json=p).json() for p in rows]
    np.testing.assert_allclose([b["churn_probability"] for b in batch],
                               [s["churn_probability"] for s in singles], atol=1e-9)


@pytest.mark.parametrize("n,status", [(0, 422), (5001, 422)])
def test_batch_size_bounds(client, raw_df, n, status):
    one = payload(raw_df.iloc[0])
    res = client.post("/score/batch", json=[one] * n)
    assert res.status_code == status


def test_batch_rejects_a_non_list(client):
    assert client.post("/score/batch", json={"customers": []}).status_code == 422
