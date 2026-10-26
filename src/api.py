"""FastAPI single-customer scoring endpoint (Phase 10).

Run from the project root:
    python -m uvicorn src.api:app --port 8000
Then:
    POST /score  {"customerID": "...", "tenure": 12, ...}   (any subset -> defaults)
    GET  /health

The endpoint scores with outputs/model_calibrated.pkl, the same artifact the batch
job and the dashboard use, so an API response and a nightly batch row cannot disagree.
"""
from __future__ import annotations

import sys
import warnings
from typing import Literal
from pathlib import Path

warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "outputs" / "model_calibrated.pkl"
CAT, NUM = F.model_columns()

app = FastAPI(title="Churn Scoring API", version="1.1.0",
              description="Calibrated churn risk for one Telco-format customer.")

_model = None
_model_name = "model_calibrated.pkl"


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"{MODEL_PATH.name} missing - run src/run_pipeline.py")
        _model = joblib.load(MODEL_PATH)
    return _model


class Customer(BaseModel):
    """Literal domains, not just examples: OneHotEncoder runs with
    handle_unknown='ignore', so an out-of-domain category would silently encode to an
    all-zero row and score as a confident, wrong number. Rejecting at the boundary is
    the only way a typo cannot become a business decision."""

    customerID: str = Field("0000-XXXX", pattern=r"^[0-9A-Za-z-]{4,12}$")
    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: Literal["No", "Yes"] = "No"
    Dependents: Literal["No", "Yes"] = "No"
    tenure: int = Field(12, ge=0, le=72)
    PhoneService: Literal["No", "Yes"] = "Yes"
    MultipleLines: Literal["No", "No phone service", "Yes"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "DSL"
    OnlineSecurity: Literal["No", "No internet service", "Yes"] = "No"
    OnlineBackup: Literal["No", "No internet service", "Yes"] = "No"
    DeviceProtection: Literal["No", "No internet service", "Yes"] = "No"
    TechSupport: Literal["No", "No internet service", "Yes"] = "No"
    StreamingTV: Literal["No", "No internet service", "Yes"] = "No"
    StreamingMovies: Literal["No", "No internet service", "Yes"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["No", "Yes"] = "Yes"
    PaymentMethod: Literal["Bank transfer (automatic)", "Credit card (automatic)",
                           "Electronic check", "Mailed check"] = "Electronic check"
    MonthlyCharges: float = Field(70.0, gt=0)
    TotalCharges: float = Field(840.0, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "customerID": "7590-VHVEG", "tenure": 1, "Contract": "Month-to-month",
        "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5, "TotalCharges": 80.5}}}


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_name, "model_present": MODEL_PATH.exists(),
            "high_risk_cutoff": F.HIGH_RISK_CUTOFF, "medium_risk_cutoff": F.MEDIUM_RISK_CUTOFF,
            "n_features": len(CAT) + len(NUM)}


def _score_frame(records: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(records)
    for col in F.CAT_RAW:  # not CAT: tenure_band is derived by engineer(), never supplied
        raw[col] = raw[col].astype(str)
    engineered = F.engineer(raw)
    F.assert_model_contract(engineered)
    return engineered


@app.post("/score")
def score(c: Customer) -> dict:
    engineered = _score_frame([c.model_dump()])
    p = float(get_model().predict_proba(engineered[CAT + NUM])[:, 1][0])
    scored = F.add_scores(engineered, [p])
    row = scored.iloc[0]
    return {"customerID": c.customerID, "churn_probability": round(p, 4),
            "risk_tier": row["risk_tier"],
            "recommended_action": F.recommended_action(row),
            "revenue_at_risk_12m": round(float(row["revenue_at_risk_12m"]), 2),
            "model": _model_name}


@app.post("/score/batch")
def score_batch(customers: list[Customer]):
    if not 1 <= len(customers) <= 5000:
        raise HTTPException(status_code=422, detail="send between 1 and 5000 customers")
    engineered = _score_frame([c.model_dump() for c in customers])
    probs = np.asarray(get_model().predict_proba(engineered[CAT + NUM])[:, 1], dtype=float)
    # add_scores() returns probability, tier and value-at-risk; the action is a per-row
    # rule, so it is applied here exactly as run_pipeline.py and score_batch.py do it.
    scored = F.add_scores(engineered, probs)
    scored["recommended_action"] = scored.apply(F.recommended_action, axis=1)
    return {"n": len(scored),
            "counts_by_tier": scored["risk_tier"].value_counts().to_dict(),
            "customers": [{"customerID": r.customerID, "churn_probability": round(float(r.churn_probability), 4),
                           "risk_tier": r.risk_tier, "recommended_action": r.recommended_action}
                          for r in scored.itertuples()]}
