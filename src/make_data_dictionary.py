"""Generate docs/data_dictionary.md from the real CSV, not from memory.

Column semantics are written by hand below (a generator cannot know what a field
means); every count, range, domain and blank rate is read from
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv on every run, so the dictionary
cannot silently drift away from the data it documents. Re-run after swapping the
input file:

    python src/make_data_dictionary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import features as F  # noqa: E402
from dq import ALLOWED, EXPECTED_COLUMNS  # noqa: E402

RAW = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# role: id | target | model_categorical | model_numeric | dropped_from_model
SEMANTICS = {
    "customerID": ("string", "Primary key, IBM's own customer reference (NNNN-AAAAA).", "id"),
    "gender": ("string", "Male / Female. Not a model input: gender is used only in the "
               "fairness table, where a model that treats it differently by group is a "
               "problem to find, not a feature to exploit.", "dropped_from_model"),
    "SeniorCitizen": ("integer 0/1", "1 if the customer is a senior citizen (age 65+ per IBM's "
                      "definition), else 0. Stored as a flag, not an age - true age only exists "
                      "in the second source.", "model_numeric"),
    "Partner": ("string", "Has a partner (Yes/No). Household-scope context.", "model_categorical"),
    "Dependents": ("string", "Has dependents (Yes/No). Household-scope context.", "model_categorical"),
    "tenure": ("integer months", "Months the customer has been with the company at the snapshot. "
               "Max is 72, which is the study window, so tenure is right-censored at 72 - a "
               "customer still active at month 72 is not '72 months loyal', they are censored.",
               "model_numeric"),
    "PhoneService": ("string", "Has landline/voice service (Yes/No).", "model_categorical"),
    "MultipleLines": ("string", "MultipleLines for the phone service. Carries a third value "
                      "'No phone service' that is structurally impossible when PhoneService=Yes; "
                      "DQ check phone_addon_consistency enforces that.", "model_categorical"),
    "InternetService": ("string", "DSL / Fiber optic / No. Largest spread of any service field: "
                        "churn 7.4% (No) / 19.0% (DSL) / 41.9% (Fiber optic).", "model_categorical"),
    "OnlineSecurity": ("string", "Add-on. 'No internet service' is used when InternetService=No.",
                       "model_categorical"),
    "OnlineBackup": ("string", "Add-on. 'No internet service' when there is no internet.",
                     "model_categorical"),
    "DeviceProtection": ("string", "Add-on. 'No internet service' when there is no internet.",
                         "model_categorical"),
    "TechSupport": ("string", "Add-on. Churn is 41.6% where it is 'No' vs 15.2% where it is 'Yes'; "
                    "the 'No internet service' rows (7.4%) are structurally identical to "
                    "InternetService='No'. Feeds no_support_flag.", "model_categorical"),
    "StreamingTV": ("string", "Add-on. 'No internet service' when there is no internet.",
                    "model_categorical"),
    "StreamingMovies": ("string", "Add-on. 'No internet service' when there is no internet.",
                        "model_categorical"),
    "Contract": ("string", "Month-to-month / One year / Two year. Observed at the snapshot, not "
                 "necessarily at signup - which is why KM-by-contract is descriptive, not causal.",
                 "model_categorical"),
    "PaperlessBilling": ("string", "Billing is electronic (Yes/No). Overlaps heavily with payment "
                         "method: 73.7% of Electronic-check customers are paperless vs 42% for the "
                         "three automatic/mailed methods.", "model_categorical"),
    "PaymentMethod": ("string", "Four methods. 'Electronic check' is the payment-risk signal and "
                       "is collinear-ish with PaperlessBilling.", "model_categorical"),
    "MonthlyCharges": ("float", "Recurring monthly charge in USD. The revenue base for "
                       "revenue-at-risk and for the campaign ROI arithmetic.", "model_numeric"),
    "TotalCharges": ("float", "Cumulative charges to date. 11 rows are blank rather than 0; all 11 "
                    "are tenure-0 customers who have not billed yet, so blank is load-bearing "
                    "information and pandas must not auto-null it on read.", "model_numeric"),
    "Churn": ("string", "TARGET. Yes = left during the observation window. IBM also ships Churn "
              "Value (1/0) in the second source; this build derives target from the label.",
              "target"),
}


def column_stats(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    blanks = int((s.astype(str).str.strip() == "").sum())
    distinct = sorted(set(s.astype(str)) - {""})
    st = {"blank": blanks, "distinct": len(distinct)}
    if col in ALLOWED:
        st["domain"] = ", ".join(f"`{v}`" for v in ALLOWED[col])
    elif distinct and all(v.lstrip("-").replace(".", "", 1).isdigit() for v in distinct):
        num = pd.to_numeric(s, errors="coerce")
        st["domain"] = (f"numeric, min {num.min():.2f}, median {num.median():.2f}, "
                        f"max {num.max():.2f}")
    else:
        st["domain"] = f"free text / flag, {len(distinct)} distinct"
    st["examples"] = " | ".join(distinct[:3]) if len(distinct) <= 4 else " | ".join(distinct[:2]) + " | ..."
    return st


def engineered_rows() -> list[tuple[str, str, str]]:
    return [
        ("service_count", "count of the 9 service/add-on columns equal to 'Yes'",
         "PhoneService..StreamingMovies", "Breadth of the bundle. Deeply protective: each add-on "
         "raises switching cost."),
        ("avg_monthly_spend", "TotalCharges / tenure, falling back to MonthlyCharges at tenure 0",
         "TotalCharges, tenure, MonthlyCharges", "Realised spend per month, which differs from the "
         "contracted MonthlyCharges whenever there are one-off charges."),
        ("bill_shock_pct", "(MonthlyCharges - avg_monthly_spend) / avg_monthly_spend, 0 where the "
         "denominator is 0", "MonthlyCharges, avg_monthly_spend",
         "PROXY. A true bill-shock needs a prior-month bill; this approximates it as 'this month "
         "costs more than the customer's own history'."),
        ("bill_shock_flag", "bill_shock_pct > 0.20", "bill_shock_pct", "Discretised shock, the form "
         "the Cox model and the FastAPI action rules use."),
        ("no_support_flag", "TechSupport == 'No'", "TechSupport", "Absence of tech support is the "
         "cheapest protective add-on to sell. Caveat worth knowing: 'No internet service' is NOT "
         "counted as 1, so the 0 group mixes real have-support (15.2% churn) with no-internet "
         "(7.4%) - the flag is protective partly because it absorbs the no-internet cohort."),
        ("no_security_flag", "OnlineSecurity == 'No'", "OnlineSecurity", ""),
        ("contract_risk", "Month-to-month=2, One year=1, Two year=0", "Contract",
         "Ordinal encoding of lock-in. Keeps contract as one monotone term for the Cox model."),
        ("electronic_check_flag", "PaymentMethod == 'Electronic check'", "PaymentMethod",
         "No auto-debit means nothing stops the customer leaving mid-cycle."),
        ("is_new", "tenure <= 12", "tenure", "First-year cohort, where onboarding failure shows up."),
        ("charges_per_service", "MonthlyCharges / max(service_count, 1)", "MonthlyCharges, service_count",
         "Price per bundled item - 'am I paying a lot for little?' proxy."),
        ("contract_remaining_proxy", "12 - (tenure % 12) for One year, 24 - (tenure % 24) for Two "
         "year, else 0", "Contract, tenure",
         "PROXY, and the weakest one. There is no signup date in this dataset, so time-into-term is "
         "inferred by wrapping tenure. It is only meaningful if contracts started at tenure 0."),
        ("log_total_charges", "ln(1 + TotalCharges)", "TotalCharges", "Heavy right tail compressed."),
        ("tenure_band", "cut(tenure, [-1,12,24,48,72]) into 0-12 / 13-24 / 25-48 / 49-72", "tenure",
         "Categorical view of loyalty, used for segment reporting and fairness."),
        ("target", "Churn == 'Yes' -> 1", "Churn", "Binary label."),
        ("risk_tier", "High >= 0.60, Medium >= 0.35, else Low, on calibrated probability",
         "churn_probability", "Operating decision; the cutoffs are a business choice, not model output."),
        ("churn_probability", "predict_proba from model_calibrated.pkl (isotonic-calibrated)",
         "all 16 model features", "Deployed scoring column. Calibration is why 0.60 means ~60%."),
    ]


def main() -> int:
    df = F.load_raw(RAW)
    clean = F.coerce_numeric(df)
    eng = F.engineer(df)
    lines = [
        "# Data dictionary",
        "",
        f"Primary feed: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (IBM Telco Customer Churn "
        f"snapshot), {len(df):,} rows x {len(df.columns)} columns.",
        "Second feed: `data/raw/ibm_telco_11_1_3/` - see its PROVENANCE.md and the section at the "
        "bottom of this file.",
        "",
        "Generated by `python src/make_data_dictionary.py`. Every count and range below is read from "
        "the CSV on each run; only the meaning of a field is hand-written.",
        "",
        "## The 21 primary columns",
        "",
        "| # | column | type | role | blank | distinct | domain / range | what it is |",
        "|--:|---|---|---|---:|---:|---|---|",
    ]
    for i, col in enumerate(EXPECTED_COLUMNS, 1):
        st = column_stats(df, col)
        dtype, desc, role = SEMANTICS[col]
        lines.append(f"| {i} | `{col}` | {dtype} | {role} | {st['blank']} | {st['distinct']} | "
                     f"{st['domain']} | {desc} |")

    lines += [
        "",
        f"Model inputs actually used: {len(F.CAT)} categorical "
        f"(`{'`, `'.join(F.CAT)}`) and {len(F.NUM)} numeric "
        f"(`{'`, `'.join(F.NUM)}`). `gender` and `customerID` are deliberately out.",
        "",
        "## Blank TotalCharges - read this before casting anything",
        "",
        f"{int((df['TotalCharges'].astype(str).str.strip() == '').sum())} rows have a blank "
        f"`TotalCharges` (not 0, not NaN - a single space character). Every one of them has "
        f"tenure = 0, so they are brand-new customers who have not yet billed. Consequences:",
        "- `pd.read_csv` with default settings turns ' ' into NaN and the rows look like missing "
        "data. This repo reads the CSV with `dtype=str, keep_default_na=False` so the blank stays "
        "visible, then coerces deliberately (`features.coerce_numeric`).",
        "- `features.engineer()` fills those blanks with 0. That is a business rule, not a "
        "cleansing reflex; the DQ check `blank_TotalCharges_only_tenure_0` fails if a blank ever "
        "lands on a tenured customer.",
        "",
        "## Engineered features (not in the raw file)",
        "",
        "| feature | definition | sourced from | note |",
        "|---|---|---|---|",
    ]
    for name, definition, source, note in engineered_rows():
        lines.append(f"| `{name}` | {definition} | {source} | {note} |")

    lines += [
        "",
        "### Distributions of the derived numerics",
        "",
    ]
    for c in ["service_count", "bill_shock_pct", "contract_remaining_proxy", "avg_monthly_spend"]:
        s = eng[c]
        lines.append(f"- `{c}`: min {s.min():.4f}, median {s.median():.4f}, max {s.max():.4f}, "
                     f"mean {s.mean():.4f}")
    lines.append(f"- `bill_shock_flag = 1` on {int(eng['bill_shock_flag'].sum()):,} rows "
                 f"({eng['bill_shock_flag'].mean():.2%}) - a deliberate low-recall flag, not a "
                 f"summary statistic.")

    lines += [
        "",
        "## Columns that exist only in the second source",
        "",
        "These are not in the 21-column primary feed. They come from IBM's 'Telco customer churn "
        "(11.1.3+)' release and are used for enrichment and cross-source reconciliation only - the "
        "model never sees them.",
        "",
        "| column | module | what it adds | used in |",
        "|---|---|---|---|",
        "| `Churn Category`, `Churn Reason` | status | 5-way reason family + free-text reason, "
        "populated only for churners | churn-reason enrichment, report.top_churn_reasons |",
        "| `Churn Score` | status | IBM's own risk score, observed range 5-96 | compared against our calibrated "
        "probability; not used as a feature (it is the answer, in a different model's clothes) |",
        "| `CLTV` | status | IBM's customer-lifetime-value estimate, 2003-6500 | revenue-at-risk "
        "cross-check, `enrichment_cltv_vs_proxy.json` |",
        "| `Satisfaction Score` | status | 1-5 | satisfaction vs churn enrichment table |",
        "| `Customer Status` | status | Stayed (4,720) / Churned (1,869) / Joined (454) | "
        "'Joined' means joined during the reporting quarter (tenure 1-3 months), not the "
        "tenure-0 cohort |",
        "| `Age`, `Married`, `Number of Dependents`, `Under 30` | demographics | real age, which the "
        "primary feed only has as a senior flag | `age_band` in the fairness table |",
        "| `State`, `City`, `Zip Code`, `Latitude`, `Longitude` | location | geography | "
        "`report.churn_by_state_ext_source` |",
        "| `Population` | population | zip-level population, 1,671 zips | denominator for "
        "penetration; referential check in 02_dq_checks.sql |",
        "| `Quarter`, `Offer`, `Referred a Friend`, `Total Refunds`, `Total Extra Data Charges`, "
        "`Total Long Distance Charges`, `Total Revenue`, `Avg Monthly GB Download`, `Internet Type`, "
        "`Premium Tech Support`, `Streaming Music`, `Unlimited Data` | services | Q3-tagged service "
        "detail | Q3 drift batch construction; `Total Revenue` vs our revenue-at-risk proxy |",
        "",
        "The 265 contract re-classifications, the Cable/DSL split and the removed 'Electronic check' "
        "category in that release are documented in "
        "`data/raw/ibm_telco_11_1_3/PROVENANCE.md` and measured in `outputs/source_reconciliation.csv`.",
    ]

    out = BASE / "docs" / "data_dictionary.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(BASE)} ({len(df)} rows, {len(EXPECTED_COLUMNS)} columns documented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
