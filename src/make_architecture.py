"""Regenerate docs/architecture.png from the repo as it actually is.

The diagram used to be hand-drawn, which meant it kept describing a pipeline that no
longer existed once the calibrated model, the second source and the R run landed. Box
contents are therefore read from the tree (file names, dashboard pages, DQ check
count), so a re-run after a real change shows that change:

    python src/make_architecture.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

matplotlib.rcParams["text.parse_math"] = False

BASE = Path(__file__).resolve().parents[1]

W = 14.0  # width in inches; the height is derived from the drawn content
HEADER = "#14283f"
STAGE_BG = "#e8eef5"
BOX_BG = "#f7fafd"
BOX_EDGE = "#b9c9d8"
ACCENT = "#1f6fa8"
BODY = "#2c2c2c"
MUTED = "#63707e"


def repo_facts() -> dict:
    import pandas as pd

    dq = pd.read_csv(BASE / "outputs" / "dq_scorecard.csv")
    app = (BASE / "dashboard" / "app.py").read_text(encoding="utf-8")
    block = re.search(r"^PAGES = \{(.*?)^\}", app, flags=re.S | re.M)
    pages = re.findall(r'"[^"]+":\s*"([^"]+)"', block.group(1)) if block else []
    models = pd.read_csv(BASE / "outputs" / "model_comparison.csv")["model"].tolist()
    raw = BASE / "data" / "raw"
    primary = next(p.name for p in raw.glob("*.csv"))
    sql = sorted(p.name for p in (BASE / "sql").glob("*.sql"))
    n_xlsx = len(list((raw / "ibm_telco_11_1_3").glob("*.xlsx")))
    return {
        "dq": f"{int(dq['passed'].sum())}/{len(dq)} checks",
        "pages": pages,
        "models": models,
        "primary": primary,
        "sql": sql,
        "n_xlsx": n_xlsx,
        "artefacts": len(list((BASE / "outputs").glob("*.csv"))) + len(list((BASE / "outputs").glob("*.json"))),
    }


def wrap(text: str, chars: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=chars) or [""]


V_GAP = 0.34  # inches between boxes inside a column
BOX_PAD = 0.56  # title line + bottom padding


def _box_h(lines: list[str], size: float) -> float:
    return BOX_PAD + len(lines) * size * 1.45 / 72


def _stack_h(boxes: list[tuple[str, list[str], float]]) -> float:
    return sum(_box_h(lines, size) for _, lines, size in boxes) + V_GAP * (len(boxes) - 1)


def box(ax, x, y, w, h, title, lines, *, fill=BOX_BG, edge=BOX_EDGE, title_colour=HEADER, size=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.07",
                                linewidth=1.0, edgecolor=edge, facecolor=fill))
    ax.text(x + 0.13, y + h - 0.20, title, fontsize=8.4, color=title_colour, fontweight="bold", va="top")
    ax.text(x + 0.13, y + h - 0.44, "\n".join(lines), fontsize=size, color=BODY, va="top", linespacing=1.45)


def stack(ax, x, top_y, w, boxes) -> float:
    """Draw a column of content-sized boxes downwards; returns the y of the lowest edge."""
    y = top_y
    for title, lines, size in boxes:
        h = _box_h(lines, size)
        box(ax, x, y - h, w, h, title, lines, size=size)
        y -= h + V_GAP
    return y + V_GAP


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=ACCENT, shrinkA=0, shrinkB=0))


def draw(f: dict) -> plt.Figure:
    models = "; ".join(m.replace(" (Optuna-tuned)", " (Optuna*)") for m in f["models"])
    pages = ", ".join(p.lower() for p in f["pages"])
    columns = [
        [
            ("Primary feed",
             wrap(f["primary"], 30) + ["7,043 rows x 21 columns", "IBM Telco snapshot, one row", "per customer, no dates"], 7.2),
            ("Second source (11.1.3)",
             [f"{f['n_xlsx']} xlsx modules, 7,043 keys", "age, CLTV, churn reason and", "category, satisfaction, GB,", "offer - joined 100% on ID"], 7.4),
            ("SQL staging",
             [f"sql/ ({len(f['sql'])} files)", "DuckDB: types, domains,", "segment and tier reports"], 7.4),
        ],
        [
            ("Data quality",
             [f"{f['dq']} pass", "Pandera schema: dtypes,", "ranges, domains, nulls,", "plus duplicates and", "referential checks"], 7.4),
            ("Feature engineering",
             ["service_count, avg spend,", "bill shock, contract risk,", "contract-remaining proxy,", "charges per service", "Leakage audit: no post-churn", "field reaches the model"], 7.4),
        ],
        [
            ("Model zoo",
             wrap(models, 33) + ["", "class weights, stratified CV", "(seed 42), Optuna* = 15 trials,", "threshold sweep"], 7.2),
            ("Calibration + selection",
             ["isotonic scaling on a held-out", "validation split", "deployed artefact:", "model_calibrated.pkl", "Brier 0.1621 -> 0.1336"], 7.4),
        ],
        [
            ("Explainability",
             ["SHAP on the tuned LightGBM", "and linear SHAP on the", "deployed model, PDP,", "permutation importance"], 7.4),
            ("Survival (R)",
             ["KM by contract and payment,", "log-rank, Cox PH, cox.zph,", "cross-checked vs lifelines"], 7.4),
            ("Fairness + economics",
             ["selection, recall and calibration", "gap per segment", "profit curve by threshold"], 7.4),
        ],
        [
            (f"outputs/ ({f['artefacts']} files)",
             ["csv + json + pkl written by", "run_pipeline.py, R and SQL"], 7.4),
            ("Streamlit dashboard",
             wrap(f"{len(f['pages'])} pages: {pages}", 42) + ["reads outputs/ only and", "never retrains"], 7.2),
            ("Serving + monitoring",
             ["score_batch.py (weekly),", "api.py (single customer),", "monitor.py: score + per-feature", "PSI, trigger > 0.25"], 7.4),
        ],
    ]
    limits = (
        "Honest limits:  snapshot data, so a stratified split rather than a time-based one;  "
        "no support-call, usage or complaint field exists in either feed, so bill shock, contract-remaining and RFM "
        "are proxies;  savings are scenarios at an assumed 30% save rate, not measured uplift;  KM by contract is "
        "descriptive, and the proportional-hazards test rejects for contract risk."
    )
    limit_lines = wrap(limits, 186)

    # Height follows the content: a hardcoded canvas left a two-inch gap under short columns.
    header_h, band_h, foot_h = 1.00, 0.42, 0.30 + 0.30 + len(limit_lines) * 8.2 * 1.5 / 72 + 0.34
    h_total = header_h + band_h + 0.16 + _max_stack(columns) + 0.44 + foot_h

    fig = plt.figure(figsize=(W, h_total), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, h_total)
    ax.axis("off")

    ax.text(0.45, h_total - 0.36, "Customer churn and retention platform - how the build fits together",
            fontsize=15, color=HEADER, fontweight="bold", va="top")
    ax.text(0.45, h_total - 0.78,
            "One pipeline writes every artefact; the notebook, the dashboard, the deck and the batch scorer all read "
            "the same files from outputs/.",
            fontsize=9.2, color=MUTED, va="top")

    stages = ["1  INGEST", "2  QUALITY + FEATURES", "3  MODEL", "4  EXPLAIN + SURVIVE", "5  DELIVER + WATCH"]
    col_w = 2.55
    gap = (W - 0.9 - len(stages) * col_w) / (len(stages) - 1)
    top = h_total - header_h
    body_top = top - band_h - 0.16

    for i, name in enumerate(stages):
        x = 0.45 + i * (col_w + gap)
        ax.add_patch(FancyBboxPatch((x, top - band_h), col_w, band_h,
                                    boxstyle="round,pad=0.0,rounding_size=0.07",
                                    linewidth=0, facecolor=HEADER))
        ax.text(x + col_w / 2, top - band_h / 2 - 0.01, name, fontsize=9.6, color="white",
                fontweight="bold", ha="center", va="center")
        if i:
            arrow(ax, x - gap + 0.04, top - band_h / 2, x - 0.06, top - band_h / 2)

    for i, boxes in enumerate(columns):
        stack(ax, 0.45 + i * (col_w + gap), body_top, col_w, boxes)
        if i:
            y = body_top - 0.55
            arrow(ax, 0.45 + (i - 1) * (col_w + gap) + col_w + 0.02, y, 0.45 + i * (col_w + gap) - 0.05, y)

    ax.add_patch(FancyBboxPatch((0.45, 0.32), W - 0.9, foot_h - 0.32,
                                boxstyle="round,pad=0.0,rounding_size=0.07",
                                linewidth=1.0, edgecolor=BOX_EDGE, facecolor=STAGE_BG))
    ax.text(0.62, foot_h - 0.18, "\n".join(limit_lines), fontsize=8.2, color=BODY, va="top", linespacing=1.5)
    ax.text(0.62, 0.52,
            "Reproduce: python src/run_pipeline.py  |  Rscript r/survival_analysis.R  |  python src/run_sql.py  |  "
            "streamlit run dashboard/app.py  |  python src/make_deck.py  |  python src/make_architecture.py",
            fontsize=8.0, color=MUTED, va="top")
    return fig


def _max_stack(columns) -> float:
    return max(_stack_h(boxes) for boxes in columns)


def main() -> int:
    facts = repo_facts()
    print(json.dumps({k: v for k, v in facts.items() if not isinstance(v, list)}, default=str))
    fig = draw(facts)
    out = BASE / "docs" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
