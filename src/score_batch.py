"""Batch scoring (Phase 10).

Reads a raw Telco-format CSV (or the mapped IBM 11.1.3 Q3 batch), engineers the
SAME features as the training pipeline via src/features.py, scores with the
CALIBRATED model outputs/model_calibrated.pkl, and writes the scored batch plus a
drift report against the reference snapshot.

Usage:
  python src/score_batch.py                          # re-score the raw snapshot
  python src/score_batch.py my_new.csv               # any Telco-format file
  python src/score_batch.py --ibm-q3                 # the second-source Q3 batch
  python src/score_batch.py --held-out               # the never-trained-on test split
  python src/score_batch.py my.csv out.csv --uncalibrated
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
import monitor

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
PROC = BASE / "data" / "processed"
CAT, NUM = F.model_columns()


def load_scoring_model(base: Path = BASE, calibrated: bool = True):
    name = "model_calibrated.pkl" if calibrated else "model_best.pkl"
    p = base / "outputs" / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run python src/run_pipeline.py first")
    return joblib.load(p), name


def held_out_frame() -> pd.DataFrame:
    """The stratified test split, scored row-for-row by the pipeline and never used in
    training or calibration. Same period as training, so this is a same-period control,
    not a drift signal."""
    scored = pd.read_csv(OUT / "scored_customers.csv")
    if "split" not in scored:
        raise KeyError("outputs/scored_customers.csv has no split column; rerun run_pipeline.py")
    return scored[scored["split"] == "test"].copy()


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    engineered = F.engineer(df) if "service_count" not in df.columns else df.copy()
    F.assert_model_contract(engineered)
    probs = model.predict_proba(engineered[CAT + NUM])[:, 1]
    return F.add_scores(engineered, probs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="raw Telco-format CSV")
    ap.add_argument("output", nargs="?", help="where to write the scored batch")
    ap.add_argument("--ibm-q3", action="store_true", help="score the mapped IBM 11.1.3 Q3 batch")
    ap.add_argument("--held-out", action="store_true", help="score the held-out test split (control batch)")
    ap.add_argument("--uncalibrated", action="store_true", help="score with model_best.pkl instead of the "
                                                                "calibrated model")
    ap.add_argument("--no-drift", action="store_true", help="skip writing the drift report")
    args = ap.parse_args(argv)

    model, model_name = load_scoring_model(calibrated=not args.uncalibrated)
    if args.held_out:
        batch = held_out_frame()
        name, out_path = "held_out_test_split", PROC / "scored_heldout_batch.csv"
    elif args.ibm_q3:
        import external
        raw = F.load_raw(BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        q3 = external.q3_to_telco_schema(external.load_ibm_modules(BASE))
        batch = score(q3, model)
        name, out_path = "ibm_q3_rerelease", PROC / "scored_batch_q3.csv"
    else:
        inp = Path(args.input) if args.input else BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        batch = score(F.load_raw(inp), model)
        name = Path(inp).stem
        out_path = Path(args.output) if args.output else PROC / "scored_batch.csv"

    batch["scored_with_model"] = model_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(out_path, index=False, lineterminator="\n")
    print(f"scored {len(batch)} customers with {model_name} -> {out_path}")
    print("risk tiers:", batch["risk_tier"].value_counts().to_dict())

    if not args.no_drift:
        ref = pd.read_csv(OUT / "scored_customers.csv")
        report = monitor.build_drift_report(ref, {name: batch}, NUM, CAT)
        report.to_csv(OUT / f"drift_report_{name}.csv", index=False, lineterminator="\n")
        print(report[report["measurement"].isin(["score_psi", "realised_roc_auc", "realised_pr_auc",
                                                 "batch_churn_rate"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
