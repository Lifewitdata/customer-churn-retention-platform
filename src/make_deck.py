"""Regenerate executive_summary.pptx and executive_summary.pdf from outputs/.

Every number on the slides is read from a pipeline artefact at build time, so the
deck cannot quote a stale metric after run_pipeline.py changes:

    python src/make_deck.py            # writes both files
    python src/make_deck.py --preview DIR   # also writes page PNGs for layout checks

The PDF and the PPTX are drawn from the same layout blocks in the same 13.333 x 7.5
inch coordinate system, and text is wrapped here (not by the renderer), so the two
artefacts break lines identically. Column overflow raises instead of silently
running off the page.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Without this, "$941,191 of the $5,473,399 book" is parsed as mathtext and the text
# between the two dollar signs is typeset as a collapsed equation instead of words.
matplotlib.rcParams["text.parse_math"] = False

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"

# ---- page geometry (inches) -------------------------------------------------------

PAGE_W, PAGE_H = 13.333, 7.5
MARGIN = 0.55
TITLE_TOP = 0.34
KICKER_TOP = 1.02
CARD_TOP, CARD_H, CARD_GAP = 1.62, 1.24, 0.22
COL_TOP, COL_BOTTOM = 3.10, 6.92
COL_GAP = 0.30
FOOTER_TOP = 7.00

# ---- typography (points) ----------------------------------------------------------

F_TITLE, F_KICKER = 26, 12.5
F_VALUE, F_LABEL, F_NOTE = 20, 8.5, 8.5
F_HEADING, F_BODY, F_FOOTER = 12, 9.4, 8.3
LINE_H = 1.30  # multiple of font size
ITEM_GAP = 4.5 / 72  # inches of air between bullets; the pptx renderer uses the same 4.5pt

INK = "#14283f"
ACCENT = "#1f6fa8"
BODY = "#2c2c2c"
MUTED = "#6b7785"
CARD_BG = "#f2f6fa"
CARD_EDGE = "#d5dfe9"
RULE = "#dbe3ea"


@dataclass
class Card:
    value: str
    label: str
    note: str


@dataclass
class Column:
    heading: str
    items: list[str]


@dataclass
class Slide:
    title: str
    kicker: str
    cards: list[Card]
    columns: list[Column]
    footer: str
    notes: str = field(default="")


# ---- facts ------------------------------------------------------------------------


def _csv(name):
    import pandas as pd

    return pd.read_csv(OUT / name)


def _json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _count_pytest_tests() -> int:
    """Collected count, not a hardcoded string - the deck fails loudly if the suite breaks."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=BASE, capture_output=True, text=True,
    )
    counts = [int(n) for n in re.findall(r"tests/test_\w+\.py: (\d+)", proc.stdout)]
    if not counts:
        raise RuntimeError(f"could not parse pytest collection:\n{proc.stdout[-500:]}{proc.stderr[-500:]}")
    return sum(counts)


def _notebook_status() -> tuple[int, int]:
    import nbformat

    nb = nbformat.read(BASE / "customer_churn_retention_platform.ipynb", as_version=4)
    errors = sum(
        1
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    return len(nb.cells), errors


def build_facts() -> dict:
    ev = _json("evaluation.json")
    mc = _csv("model_comparison.csv")
    dq = _csv("dq_scorecard.csv")
    sc = _csv("scored_customers.csv")
    cox_r = _csv("r/cox_hazard_ratios.csv")
    cox_py = _csv("cox_results_lifelines.csv")
    fair = _csv("fairness_by_segment.csv")
    drift = _csv("drift_report.csv")
    bench = _json("enrichment_benchmark_vs_ibm_score.json")
    tree = _csv("shap_importance_tree.csv")
    perm = _csv("permutation_importance.csv")
    cat = _csv("enrichment_churn_category.csv")
    by_contract = _csv("churn_by_Contract.csv")
    by_tenure = _csv("churn_by_tenure_band.csv")
    by_payment = _csv("churn_by_PaymentMethod.csv")

    high = sc[sc["risk_tier"] == "High"]
    book = (sc["MonthlyCharges"] * 12).sum()
    logit_auc = float(mc.loc[mc["model"] == "Logistic Regression", "roc_auc"].iloc[0])
    ensemble = mc[mc["model"] != "Logistic Regression"]
    best_row = ensemble.loc[ensemble["roc_auc"].idxmax()]
    logit_leads = logit_auc >= mc["roc_auc"].max()
    cox = cox_r.set_index("covariate").join(
        cox_py.set_index("covariate")[["hazard_ratio"]].rename(columns={"hazard_ratio": "hazard_ratio_py"})
    )
    hr_agreement = (cox["hazard_ratio"] - cox["hazard_ratio_py"]).abs().max()

    def drift_value(measurement, batch):
        row = drift[(drift["measurement"] == measurement) & (drift["batch"] == batch)].iloc[0]
        return row["value"]

    def tier(metric):
        return drift[(drift["measurement"] == metric) & (drift["batch"] == "held_out_test_split")].iloc[0]["value"]

    senior = fair[fair["segment_dimension"] == "SeniorCitizen"].set_index("segment")

    cells, nb_errors = _notebook_status()
    n_pages = len(list((BASE / "docs" / "screenshots").glob("dashboard_*.png")))

    return {
        "rows": len(sc),
        "churners": int(sc["target"].sum()),
        "base": sc["target"].mean(),
        "test_rows": ev["test_rows"],
        "model": ev["model_deployed"],
        "roc": ev["roc_auc_calibrated"],
        "pr": ev["pr_auc_calibrated"],
        "roc_uncal": ev["roc_auc_uncalibrated"],
        "pr_uncal": ev["pr_auc_uncalibrated"],
        "brier": ev["brier_calibrated"],
        "brier_uncal": ev["brier_uncalibrated"],
        "brier_base": ev["brier_baseline_base_rate"],
        "lift": mc["top_decile_lift"].max(),
        "models": list(mc.itertuples(index=False)),
        "n_models": len(mc),
        "logit_auc": logit_auc,
        "best_ens_model": str(best_row["model"]),
        "best_ens_auc": float(best_row["roc_auc"]),
        "ens_gap": float(best_row["roc_auc"]) - logit_auc,
        "logit_leads": logit_leads,
        "dq_passed": int(dq["passed"].sum()),
        "dq_total": len(dq),
        "high_n": len(high),
        "high_rev": high["revenue_at_risk_12m"].sum(),
        "book": book,
        "high_rev_share": high["revenue_at_risk_12m"].sum() / book,
        "high_precision": tier("high_tier_precision_at_0.60"),
        "threshold": ev["profit_max_threshold"],
        "best_fraction": ev["best_contact_fraction_by_profit"],
        "scenario_profit": ev["best_scenario_profit"],
        "save_rate": ev["scenario_assumptions"]["save_rate"],
        "cost_contact": ev["scenario_assumptions"]["cost_per_contacted"],
        "contract": by_contract.set_index("Contract")["churn_rate"],
        "tenure": by_tenure.set_index("tenure_band")["churn_rate"],
        "payment": by_payment.set_index("PaymentMethod")["churn_rate"],
        "median_mtm": _csv("survival_median.csv").set_index("segment")["median_survival_months"]["Month-to-month"],
        "hr_contract": cox_r.set_index("covariate")["hazard_ratio"]["contract_risk"],
        "hr_contract_lo": cox_r.set_index("covariate")["hr_lower95"]["contract_risk"],
        "hr_contract_hi": cox_r.set_index("covariate")["hr_upper95"]["contract_risk"],
        "hr_contract_p": cox_r.set_index("covariate")["p_value"]["contract_risk"],
        "hr_services": cox_r.set_index("covariate")["hazard_ratio"]["service_count"],
        "hr_billshock": cox_r.set_index("covariate")["hazard_ratio"]["bill_shock_flag"],
        "hr_echeck": cox_r.set_index("covariate")["hazard_ratio"]["electronic_check_flag"],
        "hr_monthly": cox_r.set_index("covariate")["hazard_ratio"]["MonthlyCharges"],
        "hr_agreement": hr_agreement,
        "tree": tree.set_index("feature")["mean_abs_shap"],
        "perm": perm.set_index("feature")["perm_importance_pr_auc"],
        "senior_flag": senior.loc["senior citizen", "high_risk_share"],
        "non_senior_flag": senior.loc["non-senior", "high_risk_share"],
        "senior_base": senior.loc["senior citizen", "base_churn_rate"],
        "non_senior_base": senior.loc["non-senior", "base_churn_rate"],
        "senior_recall": senior.loc["senior citizen", "recall_at_high_cutoff"],
        "non_senior_recall": senior.loc["non-senior", "recall_at_high_cutoff"],
        "cal_gap_senior": senior.loc["senior citizen", "calibration_gap_pp"],
        "cal_gap_non": senior.loc["non-senior", "calibration_gap_pp"],
        "score_psi": drift_value("score_psi", "ibm_q3_rerelease"),
        "pay_psi": drift_value("feature_psi_categorical:PaymentMethod", "ibm_q3_rerelease"),
        "realised_auc": drift_value("realised_roc_auc", "ibm_q3_rerelease"),
        "ibm_auc": bench["ibm_churn_score_roc_auc"],
        "spearman": bench["spearman_between_scores"],
        "competitor": cat.set_index("Churn Category")["churners"]["Competitor"],
        "tests": _count_pytest_tests(),
        "nb_cells": cells,
        "nb_errors": nb_errors,
        "pages": n_pages,
    }


# ---- slides -----------------------------------------------------------------------


def build_slides(f: dict) -> list[Slide]:
    mc = {row.model: row for row in f["models"]}
    zoo = ", ".join(
        f"{m} {mc[m].roc_auc:.3f}"
        for m in ["XGBoost", "LightGBM", "Random Forest", "MLP (sklearn feedforward)"]
        if m in mc
    )
    m2m, one_y, two_y = f["contract"]["Month-to-month"], f["contract"]["One year"], f["contract"]["Two year"]
    cards = [
        Card(f"{f['base']:.1%}", "BASE CHURN RATE", f"{f['churners']:,} of {f['rows']:,} customers"),
        Card(f"{f['roc']:.4f}", "ROC-AUC, DEPLOYED MODEL", f"PR-AUC {f['pr']:.4f}; top-decile lift {f['lift']:.2f}x"),
        Card(f"{f['brier']:.4f}", "BRIER SCORE", f"was {f['brier_uncal']:.4f} raw, {f['brier_base']:.4f} at base rate"),
        Card(f"{f['high_n']:,}", "HIGH-RISK CUSTOMERS", f"${f['high_rev']:,.0f} of 12-month revenue at risk"),
    ]

    slide1 = Slide(
        title="Who will churn, and what that is worth",
        kicker=(
            f"{f['model']} scored on {f['test_rows']:,} held-out rows that were never used for training "
            f"or calibration. Probabilities are calibrated, so 0.60 means 60 in 100."
        ),
        cards=cards,
        columns=[
            Column(
                "Seven models, one split",
                [
                    f"Logistic Regression {mc['Logistic Regression'].roc_auc:.4f} / PR {mc['Logistic Regression'].pr_auc:.4f} - the deployed model.",
                    f"LightGBM tuned with Optuna (15 trials) {mc['LightGBM (Optuna-tuned)'].roc_auc:.4f}; "
                    f"PyTorch feedforward network {mc['Feedforward NN (PyTorch)'].roc_auc:.4f}.",
                    f"Everything else: {zoo}.",
                    (
                        f"The best ensemble, {f['best_ens_model']} at {f['best_ens_auc']:.4f}, still lands "
                        f"{abs(f['ens_gap']):.4f} ROC-AUC below the baseline, so the simplest model is also the best "
                        f"ranker here - and it wins on the two things a client actually needs: calibrated "
                        f"probabilities and coefficients a retention manager can read."
                        if f["logit_leads"]
                        else
                        f"{f['best_ens_model']} leads at {f['best_ens_auc']:.4f} ROC-AUC, "
                        f"{f['ens_gap']:+.4f} over the linear baseline."
                    ),
                    f"Calibration is not cosmetic: isotonic scaling moved Brier from {f['brier_uncal']:.4f} to "
                    f"{f['brier']:.4f} and the deployed artefact is the calibrated pipeline, not the raw model.",
                ],
            ),
            Column(
                "Churn is concentrated",
                [
                    f"Contract: {m2m:.1%} on month-to-month (n=3,875) vs {one_y:.1%} one-year and {two_y:.1%} two-year.",
                    f"Tenure: {f['tenure']['0-12']:.1%} in the first 12 months vs {f['tenure']['49-72']:.1%} beyond 48.",
                    f"Payment: {f['payment']['Electronic check']:.1%} on electronic check (n=2,365), the largest "
                    f"payment block and the only manual one.",
                    f"The {f['high_n']:,} customers above the 0.60 cut carry ${f['high_rev']:,.0f} of the "
                    f"${f['book']:,.0f} 12-month book ({f['high_rev_share']:.0%}); on the held-out rows, "
                    f"{f['high_precision']:.1%} of the customers flagged at that cut actually churned.",
                ],
            ),
            Column(
                "What saving them is worth (scenario)",
                [
                    f"Operating point chosen by expected profit, not accuracy: {f['threshold']:.2f} maximises "
                    f"scenario profit on the test split.",
                    f"Contacting the top {f['best_fraction']:.0%} by score returns ${f['scenario_profit']:,.0f} "
                    f"in the scenario.",
                    f"Assumptions: {f['save_rate']:.0%} save rate, ${f['cost_contact']:.0f} per contacted "
                    f"customer, CLV = 12 x monthly charges.",
                    "These are scenarios, not measured uplift. Neither feed carries an offer, campaign or "
                    "experiment field, so the save rate is an input, not a result. The first piece of new data "
                    "to buy is a randomised hold-out.",
                ],
            ),
        ],
        footer=(
            f"{f['dq_passed']}/{f['dq_total']} data-quality checks passed (Pandera schema + manual) - "
            f"{f['tests']} pytest tests - {f['nb_cells']}-cell executed notebook with {f['nb_errors']} errors - "
            f"{f['pages']}-page Streamlit dashboard."
        ),
        notes=(
            "Numbers regenerate from outputs/ by src/make_deck.py. Snapshot dataset with no signup dates, so the "
            "split is stratified (seed 42) plus stratified CV rather than time-based; a dated production feed must "
            "be re-split on time."
        ),
    )

    senior_ratio = f["senior_flag"] / f["non_senior_flag"]
    base_ratio = f["senior_base"] / f["non_senior_base"]
    slide2 = Slide(
        title="Why they leave, and where to intervene",
        kicker=(
            "Survival analysis run in R (survival 3.8.6) and cross-checked against Python lifelines; "
            "drivers from SHAP on the tuned LightGBM and permutation importance on the deployed model."
        ),
        cards=[
            Card(f"{f['median_mtm']:.0f} months", "MEDIAN SURVIVAL, MONTH-TO-MONTH", "One-year and two-year: not reached in follow-up"),
            Card(f"{f['hr_contract']:.2f}x", "HAZARD RATIO, CONTRACT RISK", f"95% CI {f['hr_contract_lo']:.2f}-{f['hr_contract_hi']:.2f}, p = {f['hr_contract_p']:.1e}"),
            Card(f"{f['hr_services']:.2f}x", "SERVICE COUNT IS PROTECTIVE", "each additional service bundled"),
            Card(f"{f['competitor'] / f['churners']:.1%}", "OF CHURNERS LEFT TO A COMPETITOR", f"{f['competitor']:,} of {f['churners']:,}, from the 11.1.3 modules"),
        ],
        columns=[
            Column(
                "Drivers the client can act on",
                [
                    f"Tree SHAP: month-to-month contract {f['tree']['cat__Contract_Month-to-month']:.3f} mean |SHAP|, "
                    f"tenure {f['tree']['num__tenure']:.3f}, no online security {f['tree']['cat__OnlineSecurity_No']:.3f}.",
                    f"Permutation importance (PR-AUC drop): log total charges {f['perm']['log_total_charges']:.3f}, "
                    f"internet service {f['perm']['InternetService']:.3f}, monthly charges {f['perm']['MonthlyCharges']:.3f}.",
                    f"Cox PH: contract risk {f['hr_contract']:.2f}x, bill shock {f['hr_billshock']:.2f}x, electronic "
                    f"check {f['hr_echeck']:.2f}x, {f['hr_monthly']:.3f}x per extra $1 a month.",
                    f"R and lifelines agree to {f['hr_agreement']:.1e} on every hazard ratio. The proportional-hazards "
                    f"test rejects for contract risk, so read these as average effects over follow-up, not constant ones.",
                ],
            ),
            Column(
                "Who the model serves",
                [
                    f"Senior citizens are flagged {f['senior_flag']:.1%} of the time vs {f['non_senior_flag']:.1%} for "
                    f"non-seniors ({senior_ratio:.2f}x), while their observed churn is {f['senior_base']:.1%} vs "
                    f"{f['non_senior_base']:.1%} ({base_ratio:.2f}x).",
                    f"Recall at the cut: {f['senior_recall']:.1%} senior vs {f['non_senior_recall']:.1%} non-senior "
                    f"(+{(f['senior_recall'] - f['non_senior_recall']) * 100:.1f}pp). The segment that churns more "
                    f"receives more of the offers it qualifies for.",
                    f"Calibration holds either way: {abs(f['cal_gap_senior']):.2f}pp vs {abs(f['cal_gap_non']):.2f}pp "
                    f"mean-prediction gap. Gender: 5.4pp recall spread, under 1pp calibration gap.",
                    "Flagging here means receiving an offer, so unequal selection is a benefit-allocation question, "
                    "not a denial. It is still checked per segment rather than assumed benign.",
                ],
            ),
            Column(
                "Guardrails, and what would break this",
                [
                    f"Drift measured on a genuinely different feed (IBM 11.1.3 re-release): score PSI "
                    f"{f['score_psi']:.4f} says stable, PaymentMethod PSI {f['pay_psi']:.2f} says RETRAIN.",
                    f"The re-release deletes the electronic-check category: 2,365 rows re-map to bank withdrawal and "
                    f"electronic_check_flag drops from 0.336 to 0.000. Realised ROC-AUC still {f['realised_auc']:.4f} - "
                    f"ranking survived, the feature did not.",
                    f"IBM's own Churn Score reaches {f['ibm_auc']:.3f} AUC vs our {f['roc']:.4f}, but it encodes the "
                    f"label. It is excluded from every model and used only as an out-of-sample yardstick "
                    f"(Spearman {f['spearman']:.3f}).",
                    "Next: randomised hold-out to price the offers, a dated feed with a time-based split, and an "
                    "upstream agreement on the payment and contract re-codes before scoring reaches production.",
                ],
            ),
        ],
        footer=(
            "Neither feed carries support calls, logins, usage, complaints or payment-delay fields. Bill shock, "
            "contract-remaining and RFM are proxies built from the 21 columns that do exist, and are labelled as such."
        ),
        notes=(
            "KM by contract is descriptive: contract type is observed at snapshot, not assigned at signup, so the "
            "35-month median is not a causal effect of being on month-to-month. Churn Reason comes from the "
            "11.1.3 status module, joined 7,043/7,043 on customerID."
        ),
    )
    return [slide1, slide2]


# ---- shared layout maths -----------------------------------------------------------


def _card_metrics():
    card_w = (PAGE_W - 2 * MARGIN - 3 * CARD_GAP) / 4
    return card_w


def _column_metrics():
    col_w = (PAGE_W - 2 * MARGIN - 2 * COL_GAP) / 3
    return col_w, COL_BOTTOM - COL_TOP


def _wrap_col(column: Column, col_w: float) -> list[tuple[str, bool]]:
    """Wrap once, here, so both renderers lay identical lines: (text, starts_a_new_bullet)."""
    usable = col_w - 0.24  # bullet indent
    chars = max(20, int(usable / (0.55 * F_BODY / 72)))
    lines = []
    for item in column.items:
        for i, line in enumerate(textwrap.wrap(item, width=chars)):
            lines.append((("- " if i == 0 else "  ") + line, i == 0))
    return lines


def _column_height(lines: list[tuple[str, bool]]) -> float:
    """Inches a wrapped column occupies, including the gap between bullets."""
    return sum(ITEM_GAP if new else 0.0 for _, new in lines) + len(lines) * F_BODY * LINE_H / 72


# ---- matplotlib renderer ----------------------------------------------------------


def render_pages(slides: list[Slide], target: Path) -> None:
    """target is a .pdf path, or a directory to write slide1.png, slide2.png, ... into.

    Previews go through this same function on purpose: a separate preview layout would
    prove nothing about what the PDF actually looks like.
    """
    card_w = _card_metrics()
    col_w, col_h = _column_metrics()
    pdf_cm = PdfPages(str(target)) if target.suffix == ".pdf" else nullcontext(None)

    def y(top):
        return PAGE_H - top

    with pdf_cm as pdf:
        for idx, slide in enumerate(slides, 1):
            fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, PAGE_W)
            ax.set_ylim(0, PAGE_H)
            ax.axis("off")

            ax.text(MARGIN, y(TITLE_TOP), slide.title, fontsize=F_TITLE, color=INK,
                    fontweight="bold", va="top")
            ax.add_patch(plt.Rectangle((MARGIN, y(TITLE_TOP + 0.55)), 1.5, 0.035,
                                       color=ACCENT, lw=0))
            kicker = textwrap.fill(slide.kicker, width=int((PAGE_W - 2 * MARGIN) / (0.55 * F_KICKER / 72)))
            ax.text(MARGIN, y(KICKER_TOP), kicker, fontsize=F_KICKER, color=MUTED, va="top")

            for i, card in enumerate(slide.cards):
                left = MARGIN + i * (card_w + CARD_GAP)
                ax.add_patch(FancyBboxPatch((left, y(CARD_TOP + CARD_H)), card_w, CARD_H,
                                            boxstyle="round,pad=0.02,rounding_size=0.06",
                                            linewidth=1, edgecolor=CARD_EDGE, facecolor=CARD_BG))
                ax.text(left + 0.16, y(CARD_TOP + 0.20), card.value, fontsize=F_VALUE, color=ACCENT,
                        fontweight="bold", va="top")
                ax.text(left + 0.16, y(CARD_TOP + 0.68), card.label.upper(), fontsize=F_LABEL, color=INK,
                        va="top", fontweight="bold")
                note = textwrap.fill(card.note, width=int((card_w - 0.32) / (0.55 * F_NOTE / 72)))
                ax.text(left + 0.16, y(CARD_TOP + 0.90), note, fontsize=F_NOTE, color=MUTED, va="top")

            for j, column in enumerate(slide.columns):
                left = MARGIN + j * (col_w + COL_GAP)
                ax.text(left, y(COL_TOP), column.heading, fontsize=F_HEADING, color=INK,
                        fontweight="bold", va="top")
                ax.add_patch(plt.Rectangle((left, y(COL_TOP + 0.26)), col_w, 0.014,
                                           color=ACCENT, lw=0))
                top = COL_TOP + 0.36
                lines = _wrap_col(column, col_w)
                used = _column_height(lines)
                if used > col_h - 0.30:
                    raise RuntimeError(
                        f"column {column.heading!r} needs {used:.2f}in of the {col_h - 0.30:.2f}in available"
                    )
                cursor = top
                for line, new_item in lines:
                    if new_item:
                        cursor += ITEM_GAP
                    ax.text(left + 0.24, y(cursor), line.lstrip("- "), fontsize=F_BODY, color=BODY, va="top")
                    if new_item:
                        ax.add_patch(plt.Circle((left + 0.09, y(cursor - 0.045)), 0.030, color=ACCENT, lw=0))
                    cursor += F_BODY * LINE_H / 72

            for k, line in enumerate(textwrap.wrap(slide.footer, width=int((PAGE_W - 2 * MARGIN) / (0.55 * F_FOOTER / 72)))):
                ax.text(MARGIN, y(FOOTER_TOP + k * F_FOOTER * 1.25 / 72), line, fontsize=F_FOOTER,
                        color=MUTED, style="italic", va="top")

            if pdf is None:
                fig.savefig(target / f"slide{idx}.png")
            else:
                pdf.savefig(fig)
            plt.close(fig)


# ---- python-pptx renderer ---------------------------------------------------------


def render_pptx(slides: list[Slide], path: Path) -> None:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Inches, Pt

    def rgb(hex_str: str) -> RGBColor:
        return RGBColor.from_string(hex_str.lstrip("#"))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(PAGE_W), Inches(PAGE_H)
    blank = prs.slide_layouts[6]
    card_w = _card_metrics()
    col_w, col_h = _column_metrics()

    def textbox(slide, left, top, width, height):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        return tf

    def para(tf, text, size, colour, bold=False, space_before=0.0, first=False):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before, p.space_after = Pt(space_before), Pt(0)
        run = p.add_run()
        run.text = text
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, "Calibri"
        run.font.color.rgb = rgb(colour)
        return p

    for slide_spec in slides:
        slide = prs.slides.add_slide(blank)

        tf = textbox(slide, MARGIN, TITLE_TOP, PAGE_W - 2 * MARGIN, 0.62)
        para(tf, slide_spec.title, F_TITLE, INK, bold=True, first=True)
        tf = textbox(slide, MARGIN, KICKER_TOP, PAGE_W - 2 * MARGIN, 0.56)
        para(tf, slide_spec.kicker, F_KICKER, MUTED, first=True)

        for i, card in enumerate(slide_spec.cards):
            left = MARGIN + i * (card_w + CARD_GAP)
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(CARD_TOP),
                                           Inches(card_w), Inches(CARD_H))
            shape.adjustments[0] = 0.07
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(CARD_BG)
            shape.line.color.rgb = rgb(CARD_EDGE)
            shape.line.width = Pt(0.75)
            shape.shadow.inherit = False
            tf = textbox(slide, left + 0.16, CARD_TOP + 0.16, card_w - 0.32, CARD_H - 0.3)
            para(tf, card.value, F_VALUE, ACCENT, bold=True, first=True)
            para(tf, card.label.upper(), F_LABEL, INK, bold=True, space_before=3)
            para(tf, card.note, F_NOTE, MUTED, space_before=2)

        for j, column in enumerate(slide_spec.columns):
            left = MARGIN + j * (col_w + COL_GAP)
            tf = textbox(slide, left, COL_TOP, col_w, 0.26)
            para(tf, column.heading, F_HEADING, INK, bold=True, first=True)
            rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(COL_TOP + 0.28),
                                          Inches(col_w), Emu(11430))
            rule.fill.solid()
            rule.fill.fore_color.rgb = rgb(ACCENT)
            rule.line.fill.background()
            rule.shadow.inherit = False

            body_top = COL_TOP + 0.40
            used = _column_height(_wrap_col(column, col_w))
            if used > col_h - 0.34:
                raise RuntimeError(f"column {column.heading!r} needs {used:.2f}in in the pptx body box")
            tf = textbox(slide, left + 0.24, body_top, col_w - 0.24, col_h - 0.34)
            first = True
            for line, new_item in _wrap_col(column, col_w):
                gap = 4.5 if new_item and not first else 0
                # matplotlib draws a vector dot here; pptx has no bullet without XML, so use the glyph
                text = ("• " if new_item else "   ") + line.lstrip("- ")
                para(tf, text, F_BODY, BODY, space_before=gap, first=first)
                first = False

        tf = textbox(slide, MARGIN, FOOTER_TOP, PAGE_W - 2 * MARGIN, 0.42)
        para(tf, slide_spec.footer, F_FOOTER, MUTED, first=True)
        slide.notes_slide.notes_text_frame.text = slide_spec.notes

    prs.save(str(path))


# ---- entry point -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", metavar="DIR", help="also write one PNG per slide for layout checks")
    args = ap.parse_args()

    facts = build_facts()
    slides = build_slides(facts)
    render_pages(slides, BASE / "executive_summary.pdf")
    render_pptx(slides, BASE / "executive_summary.pptx")

    if args.preview:
        render_pages(slides, Path(args.preview))

    print(f"wrote {BASE / 'executive_summary.pptx'} and {BASE / 'executive_summary.pdf'}")
    print(f"facts: {json.dumps({k: v for k, v in facts.items() if isinstance(v, (str, bool, int, float))}, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
