"""Drift monitor tests (item 12).

The regression that matters here is a false negative: psi() bins on reference
quantiles, and a 0/1 indicator has only two distinct values, so the binning collapses
and psi() returns 0.0 however far the share moved. That made every categorical row in
the committed drift report read "0.0 STABLE" for the IBM Q3 batch - including
PaymentMethod, where a category carrying a third of the book went to zero. Categorical
drift is now measured on value shares (psi_discrete), and these tests pin both halves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import features as F
import monitor

BASE = Path(monitor.__file__).resolve().parents[1]


def test_psi_is_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, 4000)
    assert monitor.psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_flags_a_large_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 4000)
    batch = rng.normal(1.5, 1.4, 4000)
    assert monitor.psi(ref, batch) > 0.25


def test_psi_is_not_comparable_on_empty_or_non_finite_input():
    assert np.isnan(monitor.psi([], [1, 2]))
    assert np.isnan(monitor.psi([1, 2], []))
    assert monitor.psi([np.nan, np.nan, 1.0], [1.0, 1.0, 1.0]) == 0.0  # one distinct value left


def test_psi_on_a_binary_indicator_is_blind_and_that_is_documented():
    """The bug, kept as a test: 34% -> 0% reads 0.0 through psi(). If anyone routes
    categorical_drift back through psi(), this fails instead of shipping a fake STABLE."""
    ref = pd.Series(["A"] * 34 + ["B"] * 66)
    batch = pd.Series(["B"] * 100)
    assert monitor.psi((ref == "A").astype(float), (batch == "A").astype(float)) == 0.0
    assert monitor.psi_discrete(ref, batch) > 1.0


def test_psi_discrete_is_zero_for_identical_shares_and_grows_with_recoding():
    ref = pd.Series(["A"] * 30 + ["B"] * 70)
    assert monitor.psi_discrete(ref, ref) == pytest.approx(0.0, abs=1e-12)
    small = monitor.psi_discrete(ref, pd.Series(["A"] * 40 + ["B"] * 60))   # 0.0442
    large = monitor.psi_discrete(ref, pd.Series(["A"] * 60 + ["B"] * 40))   # 0.3762
    assert 0.0 < small < large
    assert monitor.verdict(large) == "RETRAIN / INVESTIGATE"


@pytest.mark.parametrize("value,expected", [
    (np.nan, "NOT COMPUTABLE"), (0.0, "STABLE"), (0.10, "STABLE"), (0.1001, "WATCH"),
    (0.25, "WATCH"), (0.2501, "RETRAIN / INVESTIGATE"), (3.39, "RETRAIN / INVESTIGATE")])
def test_verdict_boundaries_are_exclusive_at_the_top(value, expected):
    assert monitor.verdict(value) == expected


def test_feature_drift_only_reports_columns_present_in_both_frames():
    ref = pd.DataFrame({"tenure": np.arange(200), "MonthlyCharges": np.linspace(20, 100, 200)})
    batch = pd.DataFrame({"tenure": np.arange(200) + 12, "Ghost": np.zeros(200)})
    out = monitor.feature_drift(ref, batch, ["tenure", "MonthlyCharges", "Ghost"])
    assert list(out["column"]) == ["tenure"]
    # a constant +12-month offset moves ~6% of the mass across one decile edge: PSI 0.083,
    # which sits under the 0.10 watch trigger even though the mean moved by 50%
    assert 0.05 < out["psi"].iloc[0] < 0.10
    assert 11 < out["mean_shift_pct"].iloc[0] < 13  # +12 months on a 0-199 mean of 99.5
    assert set(out["verdict"]) == {"STABLE"}


def test_feature_drift_returns_an_empty_typed_frame_when_nothing_is_comparable():
    out = monitor.feature_drift(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}), ["a", "b"])
    assert out.empty and list(out.columns) == ["column", "kind", "psi", "verdict", "ref_mean",
                                               "batch_mean", "mean_shift_pct"]


def test_category_diffs_surfaces_unseen_and_dropped_values():
    ref = pd.DataFrame({"PaymentMethod": ["Electronic check"] * 5 + ["Bank transfer (automatic)"] * 5})
    batch = pd.DataFrame({"PaymentMethod": ["Email check"] * 4 + ["Bank transfer (automatic)"] * 6})
    out = monitor.category_diffs(ref, batch, ["PaymentMethod"])
    assert out["unseen_in_reference"].iloc[0] == "Email check"
    assert out["dropped_from_batch"].iloc[0] == "Electronic check"
    assert out["coverage_rate"].iloc[0] == 0.6
    assert out["share_rows_unseen"].iloc[0] == 0.4


def test_category_diffs_is_silent_when_domains_match(raw_df):
    ref = raw_df.assign(PaymentMethod=raw_df["PaymentMethod"].astype(str))
    out = monitor.category_diffs(ref, ref, F.CAT_RAW)
    assert (out["unseen_in_reference"] == "-").all() and (out["dropped_from_batch"] == "-").all()
    assert (out["coverage_rate"] == 1.0).all()


def test_realised_performance_needs_labels_and_two_classes():
    unlabelled = pd.DataFrame({"churn_probability": [0.1, 0.9, 0.4]})
    assert monitor.realised_performance(unlabelled) == {}
    single_class = pd.DataFrame({"target": [1, 1, 1], "churn_probability": [0.9, 0.4, 0.1]})
    assert monitor.realised_performance(single_class) == {}
    both = pd.DataFrame({"target": [1, 1, 0, 0, 0, 0], "churn_probability": [0.9, 0.7, 0.2, 0.55, 0.8, 0.3]})
    perf = monitor.realised_performance(both)
    assert perf["batch_rows"] == 6 and perf["batch_churn_rate"] == round(2 / 6, 4)
    # 2 x 4 = 8 positive/negative pairs; 0.8 outranks both churners' partners except 0.7 > 0.8 fails
    assert perf["realised_roc_auc"] == 0.875
    assert perf["high_tier_share_flagged"] == 0.5
    assert perf["high_tier_precision_at_0.60"] == round(2 / 3, 4)
    assert perf["realised_f1_at_0.60"] == round(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0), 4)


def test_build_drift_report_on_the_reference_itself_is_stable(scored):
    rep = monitor.build_drift_report(scored, {"same_snapshot": scored}, F.NUM, F.CAT)
    row = rep[rep["measurement"] == "score_psi"].iloc[0]
    assert row["value"] == pytest.approx(0.0, abs=1e-9) and row["verdict"] == "STABLE"
    assert not (rep["measurement"].str.startswith("category_change") & (rep["value"] > 0)).any()


def test_committed_drift_report_attributes_the_q3_domain_change():
    """The headline of item 12: score PSI stays quiet while the feed moves, and the
    report says which column moved."""
    rep = pd.read_csv(BASE / "outputs" / "drift_report.csv")
    q3 = rep[rep["batch"] == "ibm_q3_rerelease"]
    assert q3[q3["measurement"] == "score_psi"]["verdict"].iloc[0] == "STABLE"
    pm = q3[q3["measurement"] == "feature_psi_categorical:PaymentMethod"].iloc[0]
    assert pm["value"] > 1.0 and pm["verdict"] == "RETRAIN / INVESTIGATE"
    dropped = q3[q3["measurement"] == "category_change:PaymentMethod"].iloc[0]
    assert "Electronic check" in dropped["detail"]
    assert (q3[q3["measurement"].str.startswith("category_change")]["verdict"] == "SCHEMA CHANGE").all()


def test_committed_held_out_control_is_stable_everywhere():
    rep = pd.read_csv(BASE / "outputs" / "drift_report.csv")
    ho = rep[rep["batch"] == "held_out_test_split"]
    assert (ho[ho["measurement"].str.startswith(("score_psi", "feature_psi"))]["value"] < 0.10).all()
    assert not ho["measurement"].str.startswith("category_change").any()


def test_every_psi_row_carries_a_real_verdict():
    rep = pd.read_csv(BASE / "outputs" / "drift_report.csv")
    psi_rows = rep[rep["measurement"].str.startswith(("score_psi", "feature_psi"))]
    assert set(psi_rows["verdict"]) <= {"STABLE", "WATCH", "RETRAIN / INVESTIGATE", "NOT COMPUTABLE"}
    assert len(psi_rows) > 40 and psi_rows["value"].notna().all()


def test_drift_figure_is_written(tmp_path, scored):
    path = tmp_path / "drift.png"
    monitor.write_drift_figure(scored, {"copy": scored}, path)
    assert path.exists() and path.stat().st_size > 5000
