-- 01_ingest.sql -- land the primary feed and the second source into DuckDB, then
-- derive the model features in SQL so they can be cross-checked against pandas.
--
-- Executed by src/run_sql.py (there is no duckdb CLI on this machine):
--   python src/run_sql.py
-- Placeholders {raw_csv}, {q3_csv}, {pop_csv} are filled by that runner.
--
-- Deliberate choice: the raw CSV is read with all_varchar=true so DuckDB sees the
-- same strings pandas sees, including the 11 TotalCharges values that are " ".
-- Casting is then explicit, which is what makes the DQ layer in 02_dq_checks.sql
-- meaningful rather than accidental.

CREATE OR REPLACE SCHEMA stg;
CREATE OR REPLACE SCHEMA core;
CREATE OR REPLACE SCHEMA report;

-- ---- Stage 1: byte-faithful landing ----
CREATE OR REPLACE TABLE stg.telco_customer AS
SELECT * FROM read_csv('{raw_csv}', header = true, all_varchar = true);

-- Q3 batch: the IBM 11.1.3+ services/status/demographics modules re-expressed in
-- the primary 21-column schema by src/external.py (see
-- data/raw/ibm_telco_11_1_3/PROVENANCE.md for what it is and is not).
CREATE OR REPLACE TABLE stg.q3_customer AS
SELECT * FROM read_csv('{q3_csv}', header = true, all_varchar = true);

CREATE OR REPLACE TABLE stg.zip_population AS
SELECT * FROM read_csv('{pop_csv}', header = true, all_varchar = true);

-- The scored base is produced by the model, not by SQL, so it is loaded as a fact
-- source rather than re-derived here. Reporting (03) runs against this so the SQL
-- reports and the Streamlit dashboard read the same probabilities.
CREATE OR REPLACE TABLE core.scored AS
SELECT * FROM read_csv('{scored_csv}', header = true);

-- ---- Stage 2: typed customer table ----
-- TotalCharges blank -> NULL here, and only here: filling it with 0 is a business
-- rule (a tenure-0 customer has not billed yet), not a parsing accident.
CREATE OR REPLACE VIEW v_telco_cast AS
SELECT
    customerID,
    gender,
    CAST(SeniorCitizen   AS TINYINT)                          AS "SeniorCitizen",
    Partner,
    Dependents,
    CAST(tenure          AS SMALLINT)                         AS "tenure",
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    "Contract",
    PaperlessBilling,
    "PaymentMethod",
    CAST(MonthlyCharges  AS DOUBLE)                           AS "MonthlyCharges",
    TRY_CAST(NULLIF(TRIM(TotalCharges), '') AS DOUBLE)        AS "TotalCharges",
    NULLIF(TRIM(TotalCharges), '')                            AS total_charges_raw,
    "Churn"
FROM stg.telco_customer;

CREATE OR REPLACE TABLE core.customer AS
SELECT c.*,
       CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END           AS target,
       'primary'                                              AS source_feed
FROM v_telco_cast c;

CREATE OR REPLACE TABLE core.q3_customer AS
SELECT c.*,
       CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END           AS target,
       'ibm_11_1_3_q3'                                        AS source_feed
FROM (
    SELECT
        customerID, gender, CAST(SeniorCitizen AS TINYINT) AS "SeniorCitizen",
        Partner, Dependents, CAST(tenure AS SMALLINT) AS "tenure",
        PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "Contract",
        PaperlessBilling, "PaymentMethod",
        CAST(MonthlyCharges AS DOUBLE) AS "MonthlyCharges",
        TRY_CAST(NULLIF(TRIM(TotalCharges), '') AS DOUBLE) AS "TotalCharges",
        NULLIF(TRIM(TotalCharges), '') AS total_charges_raw,
        "Churn"
    FROM stg.q3_customer
) c;

-- ---- Stage 3: features, mirroring src/features.py engineer() ----
-- Every expression here has a pandas counterpart; tests/test_sql.py compares the two
-- column by column and fails on any mismatch. The chain is split in two only because
-- bill_shock and charges_per_service need avg_monthly_spend / service_count to already
-- exist, which SQL does not allow in the same SELECT.
CREATE OR REPLACE VIEW core.customer_spend_features AS
SELECT
    *,
    COALESCE("TotalCharges", 0)                                                    AS total_charges_filled,
    (("PhoneService"     = 'Yes')::INT + ("MultipleLines"   = 'Yes')::INT
   + ("InternetService" = 'Yes')::INT + ("OnlineSecurity"   = 'Yes')::INT
   + ("OnlineBackup"     = 'Yes')::INT + ("DeviceProtection" = 'Yes')::INT
   + ("TechSupport"      = 'Yes')::INT + ("StreamingTV"      = 'Yes')::INT
   + ("StreamingMovies"  = 'Yes')::INT)                                           AS service_count,
    CASE WHEN "tenure" > 0 THEN COALESCE("TotalCharges", 0) / "tenure"
         ELSE "MonthlyCharges" END                                                 AS avg_monthly_spend
FROM core.customer;

CREATE OR REPLACE VIEW core.customer_risk_features AS
SELECT
    *,
    COALESCE(("MonthlyCharges" - avg_monthly_spend) / NULLIF(avg_monthly_spend, 0), 0)
                                                                                   AS bill_shock_pct,
    ("TechSupport"     = 'No')::INT                                                AS no_support_flag,
    ("OnlineSecurity"  = 'No')::INT                                                AS no_security_flag,
    CASE "Contract" WHEN 'Month-to-month' THEN 2 WHEN 'One year' THEN 1
                    ELSE 0 END                                                     AS contract_risk,
    ("PaymentMethod" = 'Electronic check')::INT                                    AS electronic_check_flag,
    ("tenure" <= 12)::INT                                                          AS is_new,
    "MonthlyCharges" / GREATEST(service_count, 1)                                  AS charges_per_service,
    CASE "Contract" WHEN 'One year'  THEN 12 - ("tenure" % 12)
                    WHEN 'Two year'  THEN 24 - ("tenure" % 24)
                    ELSE 0 END                                                     AS contract_remaining_proxy,
    LN(1 + total_charges_filled)                                                   AS log_total_charges,
    CASE WHEN "tenure" <= 12 THEN '0-12'
         WHEN "tenure" <= 24 THEN '13-24'
         WHEN "tenure" <= 48 THEN '25-48'
         ELSE '49-72' END                                                          AS tenure_band
FROM core.customer_spend_features;

CREATE OR REPLACE TABLE core.customer_scored_features AS
SELECT
    customerID, gender, "SeniorCitizen", Partner, Dependents, "tenure",
    PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "Contract",
    PaperlessBilling, "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    target,
    service_count, avg_monthly_spend, bill_shock_pct,
    (bill_shock_pct > 0.20)::INT                                                   AS bill_shock_flag,
    no_support_flag, no_security_flag, contract_risk, electronic_check_flag,
    is_new, charges_per_service, contract_remaining_proxy, log_total_charges,
    tenure_band
FROM core.customer_risk_features;
