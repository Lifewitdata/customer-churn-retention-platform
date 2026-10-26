"""Second data source: IBM's later 'Telco customer churn (11.1.3+)' release.

The build's primary feed is the classic WA_Fn-UseC_-Telco-Customer-Churn.csv
snapshot. IBM republished the same book of business as six modules with extra
fields (geography, CLTV, Churn Reason/Category, Satisfaction Score, Age,
Quarter-tagged service records). This module loads those, reconciles them against
the primary feed, and can re-express the Q3-tagged service/status/demographics
modules in the primary 21-column schema so the same model can score them.

What it is NOT: a later time period. Verified in this build, tenure, monthly
charges and churn agree with the primary feed for ~99.8-100% of rows, so the Q3
batch is a re-coded re-release of the same snapshot. It is used as an
upstream-schema-change / re-release drift test and the labels let us measure
realised discrimination on it. It is not evidence of temporal drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IBM_DIR = "raw/ibm_telco_11_1_3"
MODULES = {
    "main": "Telco_customer_churn.xlsx",
    "services": "Telco_customer_churn_services.xlsx",
    "demographics": "Telco_customer_churn_demographics.xlsx",
    "status": "Telco_customer_churn_status.xlsx",
    "location": "Telco_customer_churn_location.xlsx",
    "population": "Telco_customer_churn_population.xlsx",
}

PAYMENT_MAP = {"Bank Withdrawal": "Bank transfer (automatic)", "Credit Card": "Credit card (automatic)",
               "Mailed Check": "Mailed check", "Electronic Check": "Electronic check"}
CONTRACT_MAP = {"Month-to-Month": "Month-to-month", "One Year": "One year", "Two Year": "Two year"}
INTERNET_MAP = {"DSL": "DSL", "Fiber Optic": "Fiber optic", "Cable": "DSL"}


def load_ibm_modules(base: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for key, fname in MODULES.items():
        p = Path(base) / "data" / IBM_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"missing second-source file {p} (see {IBM_DIR}/PROVENANCE.md)")
        out[key] = pd.read_excel(p)
    return out


def _prefixed(df: pd.DataFrame, prefix: str, key: str) -> pd.DataFrame:
    out = df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != key})
    return out.rename(columns={key: "customerID"})


def reconcile(primary: pd.DataFrame, ibm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row/key/field agreement between the primary feed and the IBM 11.1.3 modules."""
    main = _prefixed(ibm["main"], "m_", "CustomerID")
    dem = _prefixed(ibm["demographics"], "d_", "Customer ID")
    svc = _prefixed(ibm["services"], "q_", "Customer ID")
    p = primary.copy()
    p["tenure_num"] = pd.to_numeric(p["tenure"], errors="coerce")
    p["monthly_num"] = pd.to_numeric(p["MonthlyCharges"], errors="coerce")
    p["total_num"] = pd.to_numeric(p["TotalCharges"].astype(str).str.strip(), errors="coerce")
    m = p.merge(main, on="customerID", how="inner")
    s = p.merge(svc, on="customerID", how="inner")
    d = p.merge(dem, on="customerID", how="inner")
    tc_match = float((m["total_num"].round(2)
                      == pd.to_numeric(m["m_Total Charges"], errors="coerce").round(2)).mean())
    tenure_q = s["tenure_num"] != s["q_Tenure in Months"]
    contract_q = s["Contract"] != s["q_Contract"].map(CONTRACT_MAP)
    disagree = s.loc[contract_q]
    rows = [
        {"comparison": "customerID join coverage vs main", "metric": "matched rows / primary rows",
         "value": round(len(m) / len(primary), 4), "detail": f"{len(m)} of {len(primary)}"},
        {"comparison": "customerID join coverage vs services", "metric": "matched rows / primary rows",
         "value": round(len(s) / len(primary), 4), "detail": f"{len(s)} of {len(primary)}"},
        {"comparison": "tenure vs main Tenure Months", "metric": "exact match rate",
         "value": round(float((m["tenure_num"] == m["m_Tenure Months"]).mean()), 4), "detail": "integer compare"},
        {"comparison": "MonthlyCharges vs main Monthly Charges", "metric": "exact match rate",
         "value": round(float((m["monthly_num"].round(2) == m["m_Monthly Charges"].round(2)).mean()), 4),
         "detail": "2dp"},
        {"comparison": "Churn vs main Churn Label", "metric": "exact match rate",
         "value": round(float((m["Churn"] == m["m_Churn Label"]).mean()), 4), "detail": "Yes/No"},
        {"comparison": "Contract vs main Contract", "metric": "exact match rate",
         "value": round(float((m["Contract"] == m["m_Contract"]).mean()), 4), "detail": "main module is the same snapshot"},
        {"comparison": "TotalCharges vs main Total Charges", "metric": "exact match rate",
         "value": round(tc_match, 4),
         "detail": "both feeds leave TotalCharges blank for the 11 tenure-0 rows"},
        {"comparison": "SeniorCitizen vs demographics Senior Citizen", "metric": "exact match rate",
         "value": round(float((d["SeniorCitizen"].astype(str) == d["d_Senior Citizen"].map(
             {"Yes": "1", "No": "0"})).mean()), 4), "detail": "0/1 vs Yes/No; also equals Age >= 65 exactly"},
        {"comparison": "gender vs demographics Gender", "metric": "exact match rate",
         "value": round(float((d["gender"] == d["d_Gender"]).mean()), 4), "detail": ""},
        {"comparison": "Partner vs demographics Married", "metric": "exact match rate",
         "value": round(float((d["Partner"] == d["d_Married"]).mean()), 4), "detail": "field renamed, values identical"},
        {"comparison": "Dependents vs demographics Dependents", "metric": "exact match rate",
         "value": round(float((d["Dependents"] == d["d_Dependents"]).mean()), 4),
         "detail": f"{int((d['Dependents'] != d['d_Dependents']).sum())} rows disagree — genuine cross-source conflict"},
        {"comparison": "tenure vs services Tenure in Months", "metric": "exact match rate",
         "value": round(float((~tenure_q).mean()), 4),
         "detail": "the 11 mismatches are exactly the blank-TotalCharges tenure-0 rows; Q3 carries them at tenure 10"},
        {"comparison": "tenure vs services Tenure in Months", "metric": "pearson r",
         "value": round(float(s["tenure_num"].corr(s["q_Tenure in Months"])), 5), "detail": "near-identical"},
        {"comparison": "Contract vs services Contract", "metric": "match rate after documented map",
         "value": round(float((~contract_q).mean()), 4),
         "detail": f"{len(disagree)} rows disagree, all Month-to-month -> One Year/Two Year; "
                   f"churn rate among them {float(disagree['Churn'].eq('Yes').mean()):.1%} vs "
                   f"{float(s['Churn'].eq('Yes').mean()):.1%} overall"},
        {"comparison": "PaymentMethod vs services Payment Method", "metric": "match rate after documented map",
         "value": round(float((s["PaymentMethod"] == s["q_Payment Method"].map(PAYMENT_MAP)).mean()), 4),
         "detail": "Q3 has no 'Electronic check' category: all 2,365 map to Bank Withdrawal"},
        {"comparison": "InternetService vs services Internet Type", "metric": "match rate after documented map",
         "value": round(float((s["InternetService"] == s["q_Internet Type"].fillna("None").map(
             {**INTERNET_MAP, "None": "No"})).mean()), 4),
         "detail": "Q3 splits DSL into DSL + Cable (769 rows); Cable mapped to DSL"},
    ]
    return pd.DataFrame(rows)


def q3_to_telco_schema(ibm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Re-express the Q3 services + demographics + status modules in the primary 21-column schema."""
    svc = ibm["services"].rename(columns={"Customer ID": "customerID"})
    dem = ibm["demographics"].drop(columns=["Count"]).rename(columns={"Customer ID": "customerID"})
    status = ibm["status"].drop(columns=["Count"]).rename(columns={"Customer ID": "customerID"})
    d = svc.drop(columns=["Count"]).merge(dem, on="customerID", how="inner")
    d = d.merge(status[["customerID", "Churn Label", "Churn Score", "CLTV", "Churn Category",
                        "Churn Reason", "Satisfaction Score", "Customer Status"]], on="customerID", how="left")

    out = pd.DataFrame({"customerID": d["customerID"].astype(str)})
    out["gender"] = d["Gender"]
    out["SeniorCitizen"] = d["Senior Citizen"].map({"Yes": "1", "No": "0"})
    out["Partner"] = d["Married"]
    out["Dependents"] = d["Dependents"]
    out["tenure"] = d["Tenure in Months"].astype("Int64").astype(str)
    out["PhoneService"] = d["Phone Service"]
    out["MultipleLines"] = np.where(d["Phone Service"] == "No", "No phone service", d["Multiple Lines"])
    out["InternetService"] = np.where(d["Internet Service"] == "No", "No",
                                      d["Internet Type"].map(INTERNET_MAP).fillna("No"))
    no_int = out["InternetService"] == "No"
    for src, tgt in [("Online Security", "OnlineSecurity"), ("Online Backup", "OnlineBackup"),
                     ("Device Protection Plan", "DeviceProtection"), ("Premium Tech Support", "TechSupport"),
                     ("Streaming TV", "StreamingTV"), ("Streaming Movies", "StreamingMovies")]:
        out[tgt] = np.where(no_int, "No internet service", d[src])
    out["Contract"] = d["Contract"].map(CONTRACT_MAP)
    out["PaperlessBilling"] = d["Paperless Billing"]
    out["PaymentMethod"] = d["Payment Method"].map(PAYMENT_MAP).fillna(d["Payment Method"])
    out["MonthlyCharges"] = d["Monthly Charge"].astype(str)
    out["TotalCharges"] = d["Total Charges"].astype(str)
    out["Churn"] = d["Churn Label"]
    unmapped = sorted(set(out["PaymentMethod"]) - set(PAYMENT_MAP.values())) + \
        sorted(set(out["Contract"]) - set(CONTRACT_MAP.values()))
    if unmapped:
        raise ValueError(f"Q3 mapping left unmapped category values: {unmapped}")
    out.attrs["q3_extra"] = d[["customerID", "Quarter", "Offer", "Referred a Friend", "Number of Referrals",
                               "Satisfaction Score", "Customer Status", "Churn Category", "Churn Reason",
                               "CLTV", "Churn Score", "Age", "Number of Dependents", "Under 30"]]
    return out


# Fields only the 11.1.3 release carries. Analysis-only by contract: Churn Reason /
# Churn Category / Churn Score / CLTV are known after the outcome (Churn Score encodes the
# label outright), and geography, Age and Satisfaction Score never appear in the primary
# feed the model was trained on. None of these may enter features.CAT or features.NUM -
# tests/test_external.py asserts that set stays disjoint.
ENRICHMENT_ONLY = ["Age", "Number of Dependents", "City", "State", "Country", "ZipCode",
                   "Satisfaction Score", "Customer Status", "Churn Category", "Churn Reason",
                   "CLTV", "Churn Score"]


def enrichment_frame(primary_scored: pd.DataFrame, ibm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the leakage-free second-source fields onto scored customers.

    Churn Reason / Churn Category / Churn Score / CLTV are known only after the
    outcome, so they are used for post-hoc segmentation of churners, never as
    model features. Geography, Age and Satisfaction come from the other modules
    and are likewise analysis-only here.
    """
    main = ibm["main"][["CustomerID", "City", "State", "Country", "Zip Code", "CLTV"]].rename(
        columns={"CustomerID": "customerID", "Zip Code": "ZipCode"})
    dem = ibm["demographics"][["Customer ID", "Age", "Number of Dependents"]].rename(
        columns={"Customer ID": "customerID"})
    status = ibm["status"][["Customer ID", "Satisfaction Score", "Churn Category", "Churn Reason",
                            "Customer Status"]].rename(columns={"Customer ID": "customerID"})
    pop = ibm["population"].rename(columns={"Zip Code": "ZipCode"})
    out = (primary_scored.merge(main, on="customerID", how="left")
           .merge(dem, on="customerID", how="left")
           .merge(status, on="customerID", how="left")
           .merge(pop, on="ZipCode", how="left"))
    out["Satisfaction Score"] = pd.to_numeric(out["Satisfaction Score"], errors="coerce")
    return out
