"""SQL layer tests (item 9).

The warehouse is not decoration. These tests assert that the SQL that re-implements
ingestion, DQ and feature engineering agrees with pandas row-for-row, and that the
report tables reproduce the committed CSVs. If DuckDB and features.py ever diverge,
this is where it surfaces rather than in a dashboard that quietly shows different
numbers than the model report.

Run `python src/run_sql.py` first to rebuild warehouse/churn.duckdb and
outputs/sql_reports/ from sql/*.sql.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import dq
import features as F
import run_sql

BASE = Path(run_sql.__file__).resolve().parents[1]
DB = BASE / "warehouse" / "churn.duckdb"
REPORTS = BASE / "outputs" / "sql_reports"


@pytest.fixture
def con():
    """One connection per test: this DuckDB build keeps a single unfinished result per
    connection, so a session-scoped connection shared across tests poisons every test
    after the first one that leaves a result open."""
    if not DB.exists():
        pytest.fail(f"{DB} missing - run `python src/run_sql.py` before the test suite")
    connection = duckdb.connect(str(DB), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def scored_csv() -> pd.DataFrame:
    return pd.read_csv(BASE / "outputs" / "scored_customers.csv")


def tables_in(con, schema) -> set[str]:
    return set(con.execute("SELECT table_name FROM information_schema.tables "
                           "WHERE table_schema = ? ORDER BY 1", [schema]).df()["table_name"])


def test_every_layer_landed_the_full_book(con):
    for t in ("stg.telco_customer", "core.customer", "core.customer_scored_features",
              "core.scored", "core.dim_customer", "core.fact_churn_score", "core.q3_customer"):
        assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 7043, t


def test_sql_dq_card_agrees_with_pandera_and_pandas(con):
    n, problems = run_sql.check_dq_parity(con)
    assert n == 0, "\n".join(problems)


def test_sql_feature_mirror_equals_engineer(con):
    n, problems = run_sql.check_feature_parity(con)
    assert n == 0, "\n".join(problems)


def test_sql_risk_tiers_equal_the_pipeline_tiers(con):
    n, problems = run_sql.check_tier_parity(con)
    assert n == 0, "\n".join(problems)


def test_sql_dq_card_is_all_green_and_covers_the_domain_checks(con):
    card = con.sql("SELECT * FROM core.dq_scorecard").df()
    assert card["passed"].all(), card[~card["passed"]].to_string(index=False)
    assert len(card) >= 40
    assert card["detail"].notna().all()
    # the UNPIVOT domain checks are the reason the card exists at all
    domain = [c for c in card["check"] if c.endswith("_domain") or c.startswith("q3_")]
    assert len(domain) >= 5


def test_cross_source_sql_checks_are_theones_that_would_catch_a_bad_batch(con):
    """These have no pandas counterpart, so only SQL can assert them: the Q3 batch must
    carry the same keys, the same labels and no category the encoder has never seen."""
    wanted = ["q3_row_count_matches_primary", "q3_ids_all_present_in_primary",
              "q3_labels_align_with_primary", "q3_no_unseen_payment_categories",
              "customerID_pattern_q3", "zip_population_covers_all_locations"]
    card = con.sql("SELECT * FROM core.dq_scorecard").df().set_index("check")
    for c in wanted:
        assert c in card.index, f"{c} no longer emitted by sql/02_dq_checks.sql"
        assert bool(card.loc[c, "passed"]), f"{c}: {card.loc[c, 'detail']}"


def test_blanks_land_as_nulls_not_zeros(con):
    """TotalCharges arrives as " ". All-VARCHAR landing plus TRY_CAST(NULLIF(TRIM())) must
    leave exactly the 11 tenure-0 rows null, matching features.coerce_numeric()."""
    nulls = con.execute('SELECT COUNT(*) FROM core.customer WHERE "TotalCharges" IS NULL').fetchone()[0]
    assert nulls == 11
    assert con.execute('SELECT COUNT(*) FROM stg.telco_customer '
                       'WHERE NULLIF(TRIM("TotalCharges"), \'\') IS NULL').fetchone()[0] == 11


def test_risk_tier_summary_reproduces_the_scored_csv(con, scored_csv):
    sql = con.sql("SELECT * FROM report.risk_tier_summary").df().set_index("risk_tier")
    py = scored_csv.groupby("risk_tier").agg(customers=("customerID", "size"),
                                             revenue=("revenue_at_risk_12m", "sum"))
    for tier in ("High", "Medium", "Low"):
        assert int(sql.loc[tier, "customers"]) == int(py.loc[tier, "customers"])
        assert float(sql.loc[tier, "revenue_at_risk_12m"]) == pytest.approx(
            float(py.loc[tier, "revenue"]), abs=1.0)
        assert float(sql.loc[tier, "observed_churn_rate"]) == pytest.approx(
            float(scored_csv.loc[scored_csv["risk_tier"] == tier, "target"].mean()), abs=1e-3)


def test_model_performance_table_matches_model_comparison(con):
    sql = con.sql("SELECT model, roc_auc, pr_auc FROM report.model_performance").df()
    py = pd.read_csv(BASE / "outputs" / "model_comparison.csv")[["model", "roc_auc", "pr_auc"]]
    merged = sql.merge(py, on="model", suffixes=("_sql", "_py"))
    assert len(merged) == len(py) == 7, "every compared model must be queryable from SQL"
    np.testing.assert_allclose(merged["roc_auc_sql"], merged["roc_auc_py"], atol=1e-4)
    np.testing.assert_allclose(merged["pr_auc_sql"], merged["pr_auc_py"], atol=1e-4)
    assert "Feedforward NN (PyTorch)" in set(merged["model"])


def test_fairness_report_reconciles_with_pandas(con):
    fair_sql = con.sql("SELECT * FROM report.fairness_by_segment").df()
    fair_py = pd.read_csv(BASE / "outputs" / "fairness_by_segment.csv")
    assert len(fair_sql) == len(fair_py)
    assert sorted(fair_sql["segment"]) == sorted(fair_py["segment"])
    seg = lambda df: df.assign(segment=df["segment"].astype(str)).set_index("segment")
    sql_senior = seg(fair_sql[fair_sql["segment_dimension"] == "SeniorCitizen"])
    py_senior = seg(fair_py[fair_py["segment_dimension"] == "SeniorCitizen"])
    assert sorted(sql_senior.index) == sorted(py_senior.index)
    assert (sql_senior["n"].astype(int).sort_index().values == py_senior["n"].astype(int).sort_index().values).all()
    np.testing.assert_allclose(sql_senior["high_risk_share"].astype(float).sort_index().values,
                               py_senior["high_risk_share"].astype(float).sort_index().values, atol=1e-3)


def test_drift_watch_is_a_subset_of_the_measured_rows_and_flags_the_one_that_matters(con):
    """report.drift_watch is the review queue, not a copy of the table: it must agree with
    outputs/drift_report.csv row-for-row and it must contain the PaymentMethod domain
    change, which is the only measured drift above the retrain trigger."""
    watch = con.sql("SELECT batch, measurement, value, verdict, review_flag FROM report.drift_watch").df()
    measured = pd.read_csv(BASE / "outputs" / "drift_report.csv")[["batch", "measurement", "value"]]
    merged = watch.merge(measured, on=["batch", "measurement"], suffixes=("_sql", "_py"))
    assert len(merged) == len(watch), "every watched row must exist in the measured report"
    np.testing.assert_allclose(merged["value_sql"].astype(float),
                               merged["value_py"].astype(float), atol=1e-4)
    pm = watch[watch["measurement"] == "feature_psi_categorical:PaymentMethod"].iloc[0]
    assert pm["value"] > 1.0 and pm["review_flag"] == "action required"
    assert len(watch[watch["review_flag"] == "action required"]) == 2


def test_campaign_target_list_is_the_high_tier_and_nothing_else(con, scored_csv):
    list_ids = set(con.sql("SELECT customerID FROM report.campaign_target_list").df()["customerID"])
    high = set(scored_csv.loc[scored_csv["risk_tier"] == "High", "customerID"])
    assert list_ids == high


def test_exported_report_files_match_the_warehouse_tables(con):
    """No orphan CSV may survive a rebuild as stale evidence, and no table may fail to export."""
    expected = {f"{s}__{t}.csv" for s in ("core", "report")
                for t in con.execute("SELECT table_schema, table_name FROM information_schema.tables "
                                     "WHERE table_schema = ? AND table_type = 'BASE TABLE'", [s]).df()["table_name"]}
    present = {p.name for p in REPORTS.glob("*.csv")}
    assert present == expected, f"orphaned={present - expected} missing={expected - present}"
    for name in ("core__dq_scorecard.csv", "report__risk_tier_summary.csv"):
        assert (REPORTS / name).stat().st_size > 0


def test_sql_only_tables_exist_for_the_ext_source(con):
    assert {"churn_reason", "customer_cltv", "location", "drift_report",
            "fairness_by_segment", "model_comparison"} <= tables_in(con, "ext")
    assert con.execute("SELECT COUNT(*) FROM ext.churn_reason").fetchone()[0] > 0
