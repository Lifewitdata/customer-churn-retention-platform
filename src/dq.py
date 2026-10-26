"""Data-quality layer: Pandera schemas (all 21 Telco columns) + business checks.

Pandera is the only validation framework this repo uses. There is no Great
Expectations dependency and no GE claim in the docs.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from features import INTERNET_ADDONS, NUMERIC_RAW, SERVICE_COLS, coerce_numeric

ALLOWED = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Churn": ["No", "Yes"],
}
for _c in INTERNET_ADDONS:
    ALLOWED[_c] = ["No", "No internet service", "Yes"]

EXPECTED_COLUMNS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
                    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]


def telco_raw_schema() -> pa.DataFrameSchema:
    """Schema for the raw CSV as read with dtype=str, keep_default_na=False."""
    cols = {"customerID": pa.Column(str, unique=True, nullable=False,
                                    checks=pa.Check.str_matches(r"^[0-9]{4}-[A-Z]{5}$"))}
    for name, allowed in ALLOWED.items():
        cols[name] = pa.Column(str, pa.Check.isin(allowed), nullable=False)
    cols["tenure"] = pa.Column(str, pa.Check.str_matches(r"^[0-9]{1,2}$"), nullable=False)
    cols["MonthlyCharges"] = pa.Column(str, pa.Check.str_matches(r"^[0-9]+(\.[0-9]{1,2})?$"), nullable=False)
    # TotalCharges is the one column with legitimate blanks (11 tenure-0 rows) -> " " is allowed.
    cols["TotalCharges"] = pa.Column(str, nullable=False)
    return pa.DataFrameSchema(cols, ordered=False, title="telco_raw")


def telco_clean_schema() -> pa.DataFrameSchema:
    """Schema after coerce_numeric(): typed ranges on the numeric columns."""
    return pa.DataFrameSchema({
        "customerID": pa.Column(str, unique=True),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "tenure": pa.Column(int, pa.Check.in_range(0, 72)),
        "MonthlyCharges": pa.Column(float, pa.Check.greater_than(0)),
        "TotalCharges": pa.Column(float, pa.Check.greater_than_or_equal_to(0), nullable=True),
        "Churn": pa.Column(str, pa.Check.isin(["No", "Yes"])),
        "Contract": pa.Column(str, pa.Check.isin(ALLOWED["Contract"])),
        "PaymentMethod": pa.Column(str, pa.Check.isin(ALLOWED["PaymentMethod"])),
    }, strict=False, title="telco_clean")


def validate_schema(df: pd.DataFrame, schema: pa.DataFrameSchema) -> tuple[bool, str]:
    try:
        schema.validate(df, lazy=False)
        return True, f"{schema.title or 'schema'} passed on {len(df)} rows"
    except Exception as exc:
        return False, str(exc).splitlines()[0][:180]


def run_dq_checks(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full DQ battery for one Telco-format snapshot. Returns a tidy scorecard."""
    rows: list[dict] = []

    def add(name: str, passed, detail: str, layer: str = "business") -> None:
        rows.append({"check": name, "layer": layer, "passed": bool(passed), "detail": str(detail)})

    add("column_set_matches_21", list(df_raw.columns) == EXPECTED_COLUMNS,
        f"{len(df_raw.columns)} columns, missing={sorted(set(EXPECTED_COLUMNS) - set(df_raw.columns))}")
    add("row_count_gt_0", len(df_raw) > 0, len(df_raw))
    add("duplicate_customerID_0", df_raw["customerID"].duplicated().sum() == 0,
        int(df_raw["customerID"].duplicated().sum()))
    add("duplicate_rows_0", df_raw.duplicated().sum() == 0, int(df_raw.duplicated().sum()))
    for col, allowed in ALLOWED.items():
        if col not in df_raw:
            continue
        bad = sorted(set(df_raw[col].astype(str)) - set(allowed))
        add(f"domain_{col}", not bad, f"illegal={bad}" if bad else f"{len(allowed)} allowed values")

    df = coerce_numeric(df_raw)
    for col in NUMERIC_RAW:
        add(f"numeric_parse_{col}", df[col].notna().sum() > 0,
            f"{int(df[col].isna().sum())} unparseable")
    blank_tc = df_raw["TotalCharges"].astype(str).str.strip() == ""
    add("blank_TotalCharges_only_tenure_0", bool((df.loc[blank_tc, "tenure"] == 0).all()),
        f"{int(blank_tc.sum())} blanks, tenures={sorted(set(df.loc[blank_tc, 'tenure'].dropna().tolist()))}")
    add("null_rate_TotalCharges_lt_1pct", df["TotalCharges"].isna().mean() < 0.01,
        round(float(df["TotalCharges"].isna().mean()), 4))
    add("tenure_range_0_72", bool(df["tenure"].between(0, 72).all()),
        f"{df['tenure'].min()}-{df['tenure'].max()}")
    add("MonthlyCharges_positive", bool((df["MonthlyCharges"] > 0).all()),
        f"{df['MonthlyCharges'].min()}-{df['MonthlyCharges'].max()}")

    no_phone = df["PhoneService"] == "No"
    add("phone_addon_consistency", bool((df.loc[no_phone, "MultipleLines"] == "No phone service").all()
        and (df.loc[~no_phone, "MultipleLines"] != "No phone service").all()),
        f"{int(no_phone.sum())} no-phone rows")
    no_int = df["InternetService"] == "No"
    bad_int = {c: int(((df[c] == "No internet service") != no_int).sum()) for c in INTERNET_ADDONS
               if bool(((df[c] == "No internet service") != no_int).any())}
    add("internet_addon_consistency", not bad_int, bad_int or f"{int(no_int.sum())} no-internet rows")

    billed = df[df["tenure"] > 0]
    implied = billed["MonthlyCharges"] * billed["tenure"]
    dev = ((billed["TotalCharges"] - implied).abs() / implied).where(implied > 0)
    add("TotalCharges_within_25pct_of_monthly_x_tenure", float((dev <= 0.25).mean()) >= 0.99,
        f"{float((dev > 0.25).mean()):.2%} of billed rows deviate >25% (extra data / long-distance charges)")
    add("TotalCharges_never_negative", bool((df["TotalCharges"].fillna(0) >= 0).all()),
        float(df["TotalCharges"].min(skipna=True)))

    ok_raw, detail_raw = validate_schema(df_raw.astype(str), telco_raw_schema())
    add("pandera_schema_telco_raw", ok_raw, detail_raw, layer="schema")
    ok_clean, detail_clean = validate_schema(
        df[["customerID", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn",
            "Contract", "PaymentMethod"]].assign(SeniorCitizen=df["SeniorCitizen"].astype(int),
                                                 tenure=df["tenure"].astype(int)),
        telco_clean_schema())
    add("pandera_schema_telco_clean", ok_clean, detail_clean, layer="schema")

    return pd.DataFrame(rows)


def score_scorecard(dq: pd.DataFrame) -> float:
    return round(100 * float(dq["passed"].mean()), 1)
