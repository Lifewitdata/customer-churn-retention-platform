"""Feature engineering + model input contract, shared by the training pipeline,
batch scoring, the FastAPI endpoint, the SQL/DuckDB cross-check and pytest.

Single source of truth: run_pipeline.py, score_batch.py, api.py and the notebook
all call engineer() here, so a scored batch cannot silently drift from what was
trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_RAW = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

SERVICE_COLS = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies"]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_band"]

# CAT minus the derived tenure_band: the columns a caller has to supply. Anything in
# CAT but not here is produced by engineer(), so request models and coercion helpers
# must iterate this list, not CAT.
CAT_RAW = [c for c in CAT if c != "tenure_band"]

NUM = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "service_count", "avg_monthly_spend",
       "bill_shock_pct", "bill_shock_flag", "no_support_flag", "no_security_flag", "contract_risk",
       "electronic_check_flag", "is_new", "charges_per_service", "contract_remaining_proxy",
       "log_total_charges"]

HIGH_RISK_CUTOFF = 0.60
MEDIUM_RISK_CUTOFF = 0.35

CAMPAIGN_COST_PER_CUSTOMER = 40.0
SCENARIO_SAVE_RATE = 0.30
CLV_MONTHS = 12


def load_raw(path) -> pd.DataFrame:
    """Read the Telco CSV as strings, keeping blank TotalCharges visible as ''."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the four numeric columns; blank TotalCharges becomes NaN (11 rows, all tenure 0)."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    for c in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a raw Telco-format frame with the 16 model features added.

    Blank TotalCharges is filled with 0 (tenure-0 customers have not billed yet).
    bill_shock and contract_remaining are PROXIES built only from available columns.
    """
    df = coerce_numeric(df_raw)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["service_count"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"],
                                       df["MonthlyCharges"])
    df["bill_shock_pct"] = (df["MonthlyCharges"] - df["avg_monthly_spend"]) / df["avg_monthly_spend"].replace(0, np.nan)
    df["bill_shock_pct"] = df["bill_shock_pct"].fillna(0)
    df["bill_shock_flag"] = (df["bill_shock_pct"] > 0.20).astype(int)
    df["no_support_flag"] = (df["TechSupport"] == "No").astype(int)
    df["no_security_flag"] = (df["OnlineSecurity"] == "No").astype(int)
    df["contract_risk"] = df["Contract"].map({"Month-to-month": 2, "One year": 1, "Two year": 0})
    df["electronic_check_flag"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_new"] = (df["tenure"] <= 12).astype(int)
    df["charges_per_service"] = df["MonthlyCharges"] / df["service_count"].clip(lower=1)
    df["contract_remaining_proxy"] = np.where(
        df["Contract"] == "One year", 12 - (df["tenure"] % 12),
        np.where(df["Contract"] == "Two year", 24 - (df["tenure"] % 24), 0))
    df["log_total_charges"] = np.log1p(df["TotalCharges"])
    df["tenure_band"] = pd.cut(df["tenure"], TENURE_BINS, labels=TENURE_LABELS)
    if "Churn" in df.columns:
        df["target"] = (df["Churn"].astype(str) == "Yes").astype(int)
    return df


def model_columns() -> tuple[list[str], list[str]]:
    return list(CAT), list(NUM)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM)])


def assert_model_contract(X: pd.DataFrame) -> None:
    missing = [c for c in CAT + NUM if c not in X.columns]
    if missing:
        raise KeyError(f"model input is missing engineered features: {missing}")


def risk_tier(p) -> str:
    return "High" if p >= HIGH_RISK_CUTOFF else ("Medium" if p >= MEDIUM_RISK_CUTOFF else "Low")


def recommended_action(row) -> str:
    if row["risk_tier"] != "High":
        return "Monitor / standard care"
    if row["Contract"] == "Month-to-month" and row["electronic_check_flag"] == 1:
        return "Proactive support call + contract upgrade offer"
    if row["no_support_flag"] == 1:
        return "Proactive support call + tech-support bundle"
    if row["bill_shock_flag"] == 1:
        return "Discount / bill-review offer"
    if row["tenure"] <= 12:
        return "Product education + onboarding check-in"
    return "Loyalty reward offer"


def add_scores(df: pd.DataFrame, probs) -> pd.DataFrame:
    out = df.copy()
    out["churn_probability"] = np.asarray(probs, dtype=float)
    out["risk_tier"] = out["churn_probability"].apply(risk_tier)
    out["revenue_at_risk_12m"] = np.where(out["risk_tier"] == "High",
                                          out["MonthlyCharges"] * CLV_MONTHS, 0.0)
    return out


def scenario_profit(sub: pd.DataFrame) -> float:
    """Scenario profit for a contacted group: retained CLV x save rate - contact cost."""
    saved_revenue = float(sub.loc[sub["target"] == 1, "MonthlyCharges"].sum() * CLV_MONTHS * SCENARIO_SAVE_RATE)
    return saved_revenue - len(sub) * CAMPAIGN_COST_PER_CUSTOMER
