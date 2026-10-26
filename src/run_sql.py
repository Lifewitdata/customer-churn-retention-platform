"""DuckDB runner for the SQL layer, plus the pandas-vs-SQL cross-checks.

    python src/run_sql.py            # build the warehouse, run 01-03, cross-check
    python src/run_sql.py --no-parity

There is no `duckdb` CLI on this machine, so the .sql files are executed here. The
SQL is the source of truth for what runs; this script only substitutes paths and
turns the .xlsx second source into .csv, because DuckDB's built-in reader does not
open xlsx.

The point of the cross-checks is that they can FAIL. Three independent assertions
compare DuckDB output to pandas output row by row: the DQ verdicts, the 16
engineered features, and the risk-tier counts. If someone edits engineer() in
pandas and forgets the SQL (or the other way round), this exits non-zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external  # noqa: E402
import features as F  # noqa: E402
from dq import run_dq_checks  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SQL = BASE / "sql"
FEEDS = BASE / "data" / "processed" / "sql_feeds"
REPORTS = BASE / "outputs" / "sql_reports"
DB = BASE / "warehouse" / "churn.duckdb"

# ext.* tables are the second source plus the model artefacts, landed from CSV.
# The xlsx modules have to be flattened by pandas first: DuckDB cannot read xlsx
# without the spatial extension, which this build does not add.
EXT_TABLES = ["churn_reason", "customer_cltv", "location", "model_comparison",
              "fairness_by_segment", "drift_report"]


def materialize_feeds(ibm: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write the CSVs the SQL files read. Returns the paths for substitution."""
    FEEDS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    status = ibm["status"].rename(columns={"Customer ID": "customerID"})
    churners = status[status["Churn Value"] == 1]
    feeds = {}

    q3 = external.q3_to_telco_schema(ibm)
    feeds["q3_csv"] = FEEDS / "ibm_q3_customer.csv"
    q3.to_csv(feeds["q3_csv"], index=False)

    feeds["pop_csv"] = FEEDS / "zip_population.csv"
    ibm["population"].to_csv(feeds["pop_csv"], index=False)

    # ext.churn_reason is churners only: IBM leaves Churn Reason blank for stayers,
    # so an inner join on it is a churner cohort by construction, not by filter.
    feeds["churn_reason"] = FEEDS / "ext_churn_reason.csv"
    churners[["customerID", "Churn Reason", "Churn Category", "Satisfaction Score", "CLTV"]] \
        .to_csv(feeds["churn_reason"], index=False)

    feeds["customer_cltv"] = FEEDS / "ext_customer_cltv.csv"
    status[["customerID", "CLTV", "Satisfaction Score", "Churn Category", "Customer Status"]] \
        .to_csv(feeds["customer_cltv"], index=False)

    feeds["location"] = FEEDS / "ext_location.csv"
    loc = ibm["location"].rename(columns={"Customer ID": "customerID"})
    loc[["customerID", "State", "City", "Zip Code"]].to_csv(feeds["location"], index=False)

    for name, src in [("model_comparison", "model_comparison.csv"),
                      ("fairness_by_segment", "fairness_by_segment.csv"),
                      ("drift_report", "drift_report.csv")]:
        dst = FEEDS / f"ext_{name}.csv"
        pd.read_csv(BASE / "outputs" / src).to_csv(dst, index=False)
        feeds[name] = dst

    feeds["scored_csv"] = BASE / "outputs" / "scored_customers.csv"
    feeds["raw_csv"] = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    return feeds


def split_statements(text: str) -> list[str]:
    """Split a .sql file on ';' while ignoring semicolons inside line comments and
    string literals - prose like "-- a counterpart; test_sql_parity.py" otherwise
    cuts a statement in half. Executing the whole file in one call instead would
    work, but then a failure cannot be pointed at the statement that caused it."""
    out, buf, i, n = [], [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "-" and text.startswith("--", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
            continue
        if c == "'":
            end = i + 1
            while end < n:
                if text[end] == "'":
                    break
                end += 1
            buf.append(text[i:end + 1])
            i = end + 1
            continue
        if c == ";":
            if "".join(buf).strip():
                out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    if "".join(buf).strip():
        out.append("".join(buf).strip())
    return out


def run_sql_files(con: duckdb.DuckDBPyConnection, feeds: dict[str, Path]) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS ext;")
    for table in EXT_TABLES:
        path = (FEEDS / f"ext_{table}.csv").as_posix()
        con.execute(f"CREATE OR REPLACE TABLE ext.{table} AS SELECT * FROM read_csv('{path}')")

    paths = {"raw_csv": feeds["raw_csv"], "q3_csv": feeds["q3_csv"],
             "pop_csv": feeds["pop_csv"], "scored_csv": feeds["scored_csv"]}
    for name in ["01_ingest.sql", "02_dq_checks.sql", "03_reporting.sql"]:
        text = (SQL / name).read_text(encoding="utf-8")
        # str.replace, not str.format: the DQ file contains regexes with braces
        # (^[0-9]{4}-[A-Z]{5}$) that format() would read as replacement fields.
        for key, path in paths.items():
            text = text.replace("{" + key + "}", path.as_posix())
        stmts = split_statements(text)
        for stmt in stmts:
            try:
                con.execute(stmt).fetchall()
            except Exception as exc:  # name the failing statement instead of a stack trace
                raise RuntimeError(f"{name} failed on:\n{stmt[:400]}\n\n{exc}") from exc
        print(f"ran {name} ({len(stmts)} statements)")


def export_reports(con: duckdb.DuckDBPyConnection) -> list[pd.DataFrame]:
    """Persisted tables only. The two feature views in core/ are intermediate steps of the
    engineer() mirror, so exporting them would double the CSV size for no new information."""
    tables = con.sql("SELECT table_schema, table_name FROM information_schema.tables "
                     "WHERE table_schema IN ('core', 'report') AND table_type = 'BASE TABLE' "
                     "ORDER BY 1, 2").df()
    frames = []
    written = set()
    for _, r in tables.iterrows():
        df = con.sql(f"SELECT * FROM {r.table_schema}.{r.table_name}").df()
        out = REPORTS / f"{r.table_schema}__{r.table_name}.csv"
        df.to_csv(out, index=False)
        written.add(out.name)
        frames.append(df)
    # a renamed or dropped table must not leave its old CSV behind as "evidence"
    stale = [p for p in REPORTS.glob("*.csv") if p.name not in written]
    for p in stale:
        p.unlink()
    print(f"exported {len(tables)} tables -> {REPORTS}"
          + (f", removed {len(stale)} stale CSV(s): {[p.name for p in stale]}" if stale else ""))
    return frames


def check_dq_parity(con: duckdb.DuckDBPyConnection) -> tuple[int, list[str]]:
    """Every check name that exists in both engines must carry the same verdict."""
    sql_card = con.sql("SELECT \"check\", layer, passed, detail FROM core.dq_scorecard").df()
    py_card = run_dq_checks(F.load_raw(BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    merged = sql_card.merge(py_card, on="check", suffixes=("_sql", "_pandas"), how="outer")
    problems = []
    both = merged[merged["passed_sql"].notna() & merged["passed_pandas"].notna()]
    for _, r in both.iterrows():
        if bool(r["passed_sql"]) != bool(r["passed_pandas"]):
            problems.append(f"{r['check']}: sql={r['passed_sql']} ({r['detail_sql']}) "
                            f"vs pandas={r['passed_pandas']} ({r['detail_pandas']})")
    only_pandas = sorted(set(merged.loc[merged["passed_sql"].isna(), "check"]))
    only_sql = sorted(set(merged.loc[merged["passed_pandas"].isna(), "check"]))
    print(f"DQ parity: {len(both)} checks compared, {len(problems)} disagreements; "
          f"pandas-only={only_pandas}; sql-only={only_sql}")
    return len(problems), problems


def check_feature_parity(con: duckdb.DuckDBPyConnection) -> tuple[int, list[str]]:
    """The 16 SQL-derived features must equal features.engineer() column by column."""
    sql = con.sql("SELECT * FROM core.customer_scored_features").df().set_index("customerID")
    py = F.engineer(F.load_raw(BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")) \
        .set_index("customerID")
    problems = []
    for col in F.NUM + ["tenure_band", "target"]:
        if col not in sql.columns:
            problems.append(f"{col}: absent from the SQL feature table")
            continue
        a, b = sql[col], py[col].reindex(sql.index)
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            mismatch = int((a.astype(str) != b.astype(str)).sum())
            if mismatch:
                ex = a.index[(a.astype(str) != b.astype(str))][:3].tolist()
                problems.append(f"{col}: {mismatch} string mismatches, e.g. {ex}")
        else:
            diff = (pd.to_numeric(a, errors="coerce")
                    - pd.to_numeric(b, errors="coerce")).abs()
            worst = float(np.nanmax(diff.values)) if len(diff) else 0.0
            if worst > 1e-6:
                problems.append(f"{col}: max abs difference {worst:.3e}")
    print(f"feature parity: {len(F.NUM) + 2} columns compared, {len(problems)} mismatches")
    return len(problems), problems


def check_tier_parity(con: duckdb.DuckDBPyConnection) -> tuple[int, list[str]]:
    sql_counts = con.sql("SELECT risk_tier, COUNT(*) AS n FROM core.scored GROUP BY 1").df() \
        .set_index("risk_tier")["n"]
    py = pd.read_csv(BASE / "outputs" / "scored_customers.csv")["risk_tier"].value_counts()
    problems = [f"{t}: sql={sql_counts.get(t, 0)} pandas={py.get(t, 0)}"
                for t in sorted(set(sql_counts.index) | set(py.index))
                if int(sql_counts.get(t, 0)) != int(py.get(t, 0))]
    print(f"risk-tier parity: {dict(py)} vs {dict(sql_counts)} -> {len(problems)} mismatches")
    return len(problems), problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-parity", action="store_true", help="skip the pandas cross-checks")
    args = ap.parse_args()

    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = duckdb.connect(str(DB))
    feeds = materialize_feeds(external.load_ibm_modules(BASE))
    run_sql_files(con, feeds)
    export_reports(con)

    failures = 0
    if not args.no_parity:
        for fn in [check_dq_parity, check_feature_parity, check_tier_parity]:
            n, problems = fn(con)
            failures += n
            for p in problems:
                print(f"  MISMATCH {p}")

    summary = ["# SQL layer run summary", ""]
    summary.append(f"- Warehouse file: `{DB.relative_to(BASE)}`")
    summary.append(f"- SQL sources read: `{FEEDS.relative_to(BASE)}/` (xlsx flattened by pandas), "
                   f"`{feeds['raw_csv'].name}`")
    summary.append(f"- Report tables exported to `{REPORTS.relative_to(BASE)}/`")
    card = con.execute("SELECT COUNT(*), COUNT(*) FILTER (WHERE passed) FROM core.dq_scorecard").fetchone()
    summary.append(f"- DQ checks executed in SQL: {card[0]}, passed {card[1]}")
    if not args.no_parity:
        summary.append(f"- pandas-vs-SQL disagreements: {failures}")
    (REPORTS / "sql_run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    con.close()
    print(f"SQL layer done, {failures} parity mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
