"""Document-generator tests (src/make_deck.py, src/make_architecture.py).

The regression these guard is the one the deck existed to hide: executive_summary.pdf
was hand-written, so after the calibrated model, the second source and the R run landed
it kept printing the old headline numbers. The generators now read outputs/, so a stale
document is impossible unless a test like this one fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import make_architecture
import make_deck

BASE = Path(make_deck.__file__).resolve().parents[1]


@pytest.fixture()
def facts(monkeypatch):
    # build_facts() shells out to pytest to count the suite; a nested collection run
    # inside the suite is slow and self-referential, so stub it for the tests.
    monkeypatch.setattr(make_deck, "_count_pytest_tests", lambda: 0)
    return make_deck.build_facts()


def test_deck_facts_come_from_the_same_artefacts_the_pipeline_wrote(facts):
    ev = json.loads((BASE / "outputs" / "evaluation.json").read_text(encoding="utf-8"))
    sc = pd.read_csv(BASE / "outputs" / "scored_customers.csv")
    dq = pd.read_csv(BASE / "outputs" / "dq_scorecard.csv")

    assert facts["roc"] == ev["roc_auc_calibrated"]
    assert facts["brier"] == ev["brier_calibrated"]
    assert facts["test_rows"] == ev["test_rows"]
    assert facts["high_n"] == int((sc["risk_tier"] == "High").sum())
    assert facts["high_rev"] == pytest.approx(sc.loc[sc["risk_tier"] == "High", "revenue_at_risk_12m"].sum())
    assert facts["dq_passed"] == int(dq["passed"].sum()) == len(dq)


def test_r_and_python_survival_still_agree(facts):
    # The deck prints this number; if R and lifelines drift apart the slide is wrong.
    assert facts["hr_agreement"] < 1e-4


def test_slides_quote_the_live_numbers(facts):
    text = " ".join(
        " ".join([s.title, s.kicker, s.footer] + [c.value + c.label + c.note for c in s.cards]
                 + [col.heading + " ".join(col.items) for col in s.columns])
        for s in make_deck.build_slides(facts)
    )
    assert f"{facts['roc']:.4f}" in text
    assert f"{facts['high_n']:,}" in text
    assert facts["model"] in text
    assert "scenario" in text.lower()


def test_both_documents_render_without_overflow(tmp_path, facts):
    slides = make_deck.build_slides(facts)
    make_deck.render_pages(slides, tmp_path / "deck.pdf")
    make_deck.render_pptx(slides, tmp_path / "deck.pptx")
    assert (tmp_path / "deck.pdf").stat().st_size > 10_000
    assert (tmp_path / "deck.pptx").stat().st_size > 10_000


def test_architecture_diagram_reads_the_repo_as_it_is():
    info = make_architecture.repo_facts()
    shots = sorted(p.stem.replace("dashboard_", "") for p in (BASE / "docs" / "screenshots").glob("dashboard_*.png"))
    assert len(info["pages"]) == len(shots)
    assert info["pages"][0] == "Executive overview"
    assert info["dq"] == "35/35 checks"
    assert info["artefacts"] > 40


def test_architecture_diagram_draws(tmp_path):
    fig = make_architecture.draw(make_architecture.repo_facts())
    out = tmp_path / "arch.png"
    fig.savefig(out)
    assert out.stat().st_size > 50_000
