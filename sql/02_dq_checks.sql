-- 02_dq_checks.sql -- the battery src/dq.py runs, expressed in SQL.
--
-- Executed by src/run_sql.py. Output shape matches the pandas scorecard
-- (check, layer, passed, detail) and the check NAMES are identical, so the runner
-- can join the two cards and assert that DuckDB and pandas reach the same verdict
-- on every overlapping check. Two pandas checks (pandera_schema_*) have no SQL
-- equivalent; four sql_only checks are cross-source and have no pandas equivalent.
--
-- Domain checks run against stg.telco_customer (all VARCHAR) rather than the typed
-- table, because that is what dq.py validates: illegal categories must be caught
-- before a cast can hide them.

CREATE OR REPLACE TABLE core.dq_scorecard AS
SELECT * FROM (VALUES ('__placeholder__', 'business', TRUE, 'replaced below'))
       AS t("check", layer, passed, detail) WHERE 1 = 0;

-- ---- shape ----
INSERT INTO core.dq_scorecard
SELECT 'column_set_matches_21', 'business',
       COUNT(*) = 21, CAST(COUNT(*) AS VARCHAR) || ' columns landed in stg.telco_customer'
FROM information_schema.columns WHERE table_schema = 'stg' AND table_name = 'telco_customer';

INSERT INTO core.dq_scorecard
SELECT * FROM (
    SELECT 'row_count_gt_0' AS "check", 'business' AS layer, COUNT(*) > 0 AS passed,
           CAST(COUNT(*) AS VARCHAR) AS detail
    FROM stg.telco_customer
    UNION ALL
    SELECT 'duplicate_customerID_0', 'business',
           COUNT(*) = COUNT(DISTINCT customerID),
           CAST(COUNT(*) - COUNT(DISTINCT customerID) AS VARCHAR) || ' duplicate ids'
    FROM stg.telco_customer
);

-- No FROM clause here on purpose: the aggregates live in scalar subqueries, so an
-- outer "FROM stg.telco_customer" would emit one row per customer, not one check.
INSERT INTO core.dq_scorecard
SELECT 'duplicate_rows_0', 'business',
       (SELECT COUNT(*) FROM (SELECT * FROM stg.telco_customer GROUP BY ALL HAVING COUNT(*) > 1)) = 0,
       CAST((SELECT COUNT(*) FROM (SELECT * FROM stg.telco_customer GROUP BY ALL HAVING COUNT(*) > 1))
            AS VARCHAR) || ' fully duplicated rows';

-- ---- domains: one allowed-set row per column, checked in a single unpivot ----
-- Shorter and less error-prone than 16 hand-written UNIONs, and it means the
-- allowed sets here are the thing under test rather than the SQL around them.
CREATE OR REPLACE TEMP TABLE domain_map(col_name, allowed) AS
VALUES
    ('gender',            ['Female', 'Male']),
    ('SeniorCitizen',     ['0', '1']),
    ('Partner',           ['No', 'Yes']),
    ('Dependents',        ['No', 'Yes']),
    ('PhoneService',      ['No', 'Yes']),
    ('MultipleLines',     ['No', 'No phone service', 'Yes']),
    ('InternetService',   ['DSL', 'Fiber optic', 'No']),
    ('OnlineSecurity',    ['No', 'No internet service', 'Yes']),
    ('OnlineBackup',      ['No', 'No internet service', 'Yes']),
    ('DeviceProtection',  ['No', 'No internet service', 'Yes']),
    ('TechSupport',       ['No', 'No internet service', 'Yes']),
    ('StreamingTV',       ['No', 'No internet service', 'Yes']),
    ('StreamingMovies',   ['No', 'No internet service', 'Yes']),
    ('Contract',          ['Month-to-month', 'One year', 'Two year']),
    ('PaperlessBilling',  ['No', 'Yes']),
    ('PaymentMethod',     ['Bank transfer (automatic)', 'Credit card (automatic)',
                           'Electronic check', 'Mailed check']),
    ('Churn',             ['No', 'Yes']);

CREATE OR REPLACE TEMP VIEW cat_long AS
SELECT "name" AS col_name, "value" AS val
FROM (UNPIVOT stg.telco_customer
      ON gender, "SeniorCitizen", Partner, Dependents, PhoneService, MultipleLines,
         "InternetService", OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
         StreamingTV, StreamingMovies, "Contract", PaperlessBilling, "PaymentMethod", "Churn"
      INTO NAME "name" VALUE "value");

INSERT INTO core.dq_scorecard
WITH illegal AS (
    SELECT d.col_name, d.allowed,
           COUNT(*) FILTER (WHERE NOT list_contains(d.allowed, c.val)) AS n_bad,
           list(c.val) FILTER (WHERE NOT list_contains(d.allowed, c.val)) AS bad_vals
    FROM domain_map d LEFT JOIN cat_long c USING (col_name)
    GROUP BY 1, 2
)
SELECT 'domain_' || col_name, 'business', n_bad = 0,
       CASE WHEN n_bad = 0 THEN CAST(array_length(allowed) AS VARCHAR) || ' allowed values'
            ELSE CAST(n_bad AS VARCHAR) || ' illegal: ' || COALESCE(bad_vals::VARCHAR, '{}') END
FROM illegal;

-- ---- parse / range / business rules against the typed table ----
INSERT INTO core.dq_scorecard
WITH c AS (SELECT * FROM core.customer)
SELECT * FROM (
    SELECT 'numeric_parse_SeniorCitizen' AS "check", 'business' AS layer,
           COUNT(*) FILTER (WHERE "SeniorCitizen" IS NULL) = 0 AS passed,
           CAST(COUNT(*) FILTER (WHERE "SeniorCitizen" IS NULL) AS VARCHAR) || ' unparseable' AS detail
    FROM c
    UNION ALL
    SELECT 'numeric_parse_tenure', 'business', COUNT(*) FILTER (WHERE "tenure" IS NULL) = 0,
           CAST(COUNT(*) FILTER (WHERE "tenure" IS NULL) AS VARCHAR) || ' unparseable' FROM c
    UNION ALL
    SELECT 'numeric_parse_MonthlyCharges', 'business',
           COUNT(*) FILTER (WHERE "MonthlyCharges" IS NULL) = 0,
           CAST(COUNT(*) FILTER (WHERE "MonthlyCharges" IS NULL) AS VARCHAR) || ' unparseable' FROM c
    UNION ALL
    SELECT 'numeric_parse_TotalCharges', 'business',
           COUNT(*) FILTER (WHERE "TotalCharges" IS NULL) < COUNT(*),
           CAST(COUNT(*) FILTER (WHERE "TotalCharges" IS NULL) AS VARCHAR) || ' blank'
    FROM c
    UNION ALL
    -- The blanks are only benign if they land on tenure-0 rows; a blank on a
    -- 40-month customer is a real pipeline bug, so this check is directional.
    SELECT 'blank_TotalCharges_only_tenure_0', 'business',
           COUNT(*) FILTER (WHERE "TotalCharges" IS NULL AND "tenure" <> 0) = 0,
           CAST(COUNT(*) FILTER (WHERE "TotalCharges" IS NULL) AS VARCHAR) || ' blanks'
    FROM c
    UNION ALL
    SELECT 'null_rate_TotalCharges_lt_1pct', 'business',
           AVG(CASE WHEN "TotalCharges" IS NULL THEN 1.0 ELSE 0.0 END) < 0.01,
           CAST(ROUND(AVG(CASE WHEN "TotalCharges" IS NULL THEN 1.0 ELSE 0.0 END), 4) AS VARCHAR)
    FROM c
    UNION ALL
    SELECT 'customerID_pattern', 'business',
           COUNT(*) FILTER (WHERE NOT regexp_matches(customerID, '^[0-9]{4}-[A-Z]{5}$')) = 0,
           CAST(COUNT(*) FILTER (WHERE NOT regexp_matches(customerID, '^[0-9]{4}-[A-Z]{5}$')) AS VARCHAR)
           || ' non-conforming' FROM c
    UNION ALL
    SELECT 'customerID_pattern_q3', 'business',
           COUNT(*) FILTER (WHERE NOT regexp_matches(customerID, '^[0-9]{4}-[A-Z]{5}$')) = 0,
           CAST(COUNT(*) FILTER (WHERE NOT regexp_matches(customerID, '^[0-9]{4}-[A-Z]{5}$')) AS VARCHAR)
           || ' non-conforming' FROM core.q3_customer
    UNION ALL
    SELECT 'tenure_range_0_72', 'business',
           MIN("tenure") >= 0 AND MAX("tenure") <= 72,
           CAST(MIN("tenure") AS VARCHAR) || '-' || CAST(MAX("tenure") AS VARCHAR) FROM c
    UNION ALL
    SELECT 'MonthlyCharges_positive', 'business', MIN("MonthlyCharges") > 0,
           'min ' || CAST(MIN("MonthlyCharges") AS VARCHAR) FROM c
    UNION ALL
    SELECT 'TotalCharges_never_negative', 'business',
           COUNT(*) FILTER (WHERE COALESCE("TotalCharges", 0) < 0) = 0,
           CAST(MIN("TotalCharges") AS VARCHAR) FROM c
    UNION ALL
    SELECT 'churn_rate_not_degenerate', 'business',
           AVG(target) BETWEEN 0.05 AND 0.95,
           'churn rate ' || CAST(ROUND(AVG(target) * 100, 2) AS VARCHAR) || '%' FROM c
);

-- Add-on consistency: "No phone service" / "No internet service" must appear if and
-- only if the parent service is absent. This is the check that catches an upstream
-- feed that starts writing 'No' instead.
INSERT INTO core.dq_scorecard
SELECT * FROM (
    SELECT 'phone_addon_consistency' AS "check", 'business' AS layer,
           COUNT(*) FILTER (WHERE ("PhoneService" = 'No') <> ("MultipleLines" = 'No phone service')) = 0
           AS passed,
           CAST(COUNT(*) FILTER (WHERE "PhoneService" = 'No') AS VARCHAR) || ' no-phone rows, '
           || CAST(COUNT(*) FILTER (WHERE ("PhoneService" = 'No') <> ("MultipleLines" = 'No phone service'))
                   AS VARCHAR) || ' inconsistent' AS detail
    FROM core.customer
);

CREATE OR REPLACE TEMP VIEW addon_long AS
SELECT "name" AS col_name, "value" AS val, "InternetService" AS internet
FROM (UNPIVOT core.customer
      ON OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
      INTO NAME "name" VALUE "value");

INSERT INTO core.dq_scorecard
SELECT 'internet_addon_consistency', 'business',
       COUNT(*) FILTER (WHERE (internet = 'No') <> (val = 'No internet service')) = 0,
       CAST(COUNT(*) FILTER (WHERE (internet = 'No') <> (val = 'No internet service')) AS VARCHAR)
       || ' inconsistent (col x row)'
FROM addon_long;

INSERT INTO core.dq_scorecard
WITH billed AS (
    SELECT "MonthlyCharges" * "tenure" AS implied, "TotalCharges" AS actual, "tenure"
    FROM core.customer
    WHERE "tenure" > 0 AND "TotalCharges" IS NOT NULL AND "MonthlyCharges" * "tenure" > 0
)
SELECT 'TotalCharges_within_25pct_of_monthly_x_tenure', 'business',
       AVG(CASE WHEN abs(actual - implied) / implied <= 0.25 THEN 1.0 ELSE 0.0 END) >= 0.99,
       CAST(ROUND(100 * AVG(CASE WHEN abs(actual - implied) / implied > 0.25 THEN 1.0 ELSE 0.0 END), 2)
            AS VARCHAR) || '% of billed rows deviate >25%'
FROM billed;

-- ---- cross-source checks: only SQL can see both feeds at once ----
INSERT INTO core.dq_scorecard
SELECT * FROM (
    SELECT 'q3_row_count_matches_primary' AS "check", 'sql_only' AS layer,
           (SELECT COUNT(*) FROM core.q3_customer) = (SELECT COUNT(*) FROM core.customer) AS passed,
           CAST((SELECT COUNT(*) FROM core.q3_customer) AS VARCHAR) || ' q3 rows' AS detail
    UNION ALL
    SELECT 'q3_no_unseen_payment_categories', 'sql_only', COUNT(*) = 0,
           CAST(COUNT(*) AS VARCHAR) || ' payment methods outside the primary domain'
    FROM (SELECT DISTINCT "PaymentMethod" FROM core.q3_customer
          EXCEPT SELECT DISTINCT "PaymentMethod" FROM core.customer)
    UNION ALL
    SELECT 'q3_contract_domain_still_valid', 'sql_only', COUNT(*) = 0,
           CAST(COUNT(*) AS VARCHAR) || ' contract values outside the primary domain'
    FROM (SELECT DISTINCT "Contract" FROM core.q3_customer
          EXCEPT SELECT DISTINCT "Contract" FROM core.customer)
    UNION ALL
    SELECT 'q3_labels_align_with_primary', 'sql_only',
           COUNT(*) FILTER (WHERE q.target <> p.target) = 0,
           CAST(COUNT(*) FILTER (WHERE q.target <> p.target) AS VARCHAR) || ' label disagreements'
    FROM core.q3_customer q JOIN core.customer p USING (customerID)
    UNION ALL
    SELECT 'q3_ids_all_present_in_primary', 'sql_only',
           (SELECT COUNT(*) FROM (SELECT customerID FROM core.q3_customer
                                  EXCEPT SELECT customerID FROM core.customer)) = 0,
           CAST((SELECT COUNT(*) FROM (SELECT customerID FROM core.q3_customer
                                       EXCEPT SELECT customerID FROM core.customer)) AS VARCHAR)
           || ' q3 ids not in primary' AS detail
    UNION ALL
    -- Geography lives only in the second source, so the population reference table
    -- has to be checked against the location module rather than the primary feed.
    SELECT 'zip_population_covers_all_locations', 'sql_only',
           (SELECT COUNT(*) FROM (SELECT DISTINCT "Zip Code" FROM ext.location
                                  EXCEPT SELECT DISTINCT "Zip Code" FROM stg.zip_population)) = 0,
           CAST((SELECT COUNT(*) FROM (SELECT DISTINCT "Zip Code" FROM ext.location
                                       EXCEPT SELECT DISTINCT "Zip Code" FROM stg.zip_population))
                AS VARCHAR) || ' zips with no population row'
);

SELECT "check", layer, passed, detail FROM core.dq_scorecard ORDER BY layer, "check";

SELECT COUNT(*) AS total_checks,
       COUNT(*) FILTER (WHERE passed) AS passed_checks,
       ROUND(100.0 * COUNT(*) FILTER (WHERE passed) / COUNT(*), 1) AS score_pct,
       COUNT(*) FILTER (WHERE NOT passed) AS failures
FROM core.dq_scorecard;
