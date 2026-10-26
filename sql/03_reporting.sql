-- 03_reporting.sql -- the operational read-out an analyst or CRM operator would run.
--
-- Executed by src/run_sql.py after 01 and 02. It reads the scored base that the
-- pipeline wrote (outputs/scored_customers.csv, loaded as core.scored) plus the
-- second-source enrichment tables (ext.*), and materialises a small star of report
-- tables. Every number here comes from the same scored file the dashboard shows, so
-- SQL and Streamlit cannot disagree.

-- ---- dimensions ----
CREATE OR REPLACE TABLE core.dim_risk_tier AS
SELECT * FROM (VALUES
    ('High',   0.60, 1.00, 'Save now: cost of contact is well below retained CLV'),
    ('Medium', 0.35, 0.60, 'Watchlist: cheap touches only (content, usage nudges)'),
    ('Low',    0.00, 0.35, 'No intervention - contacting these customers is pure cost')
) AS t(risk_tier, prob_low, prob_high, action_guidance);

CREATE OR REPLACE TABLE core.dim_customer AS
SELECT customerID, gender, "SeniorCitizen", Partner, Dependents, "tenure",
       "Contract", "PaymentMethod", InternetService, PaperlessBilling,
       service_count, contract_risk, electronic_check_flag, no_support_flag,
       bill_shock_flag, tenure_band, is_new
FROM core.customer_scored_features;

-- ---- facts ----
CREATE OR REPLACE TABLE core.fact_churn_score AS
SELECT s.customerID,
       s."MonthlyCharges",
       COALESCE(s."TotalCharges", 0)                       AS total_charges,
       s.churn_probability,
       s.risk_tier,
       s.target                                            AS churned,
       CASE WHEN s.risk_tier = 'High' THEN s."MonthlyCharges" * 12 ELSE 0 END
                                                           AS revenue_at_risk_12m,
       CASE WHEN s.risk_tier = 'High' THEN 40.0 ELSE 0 END AS campaign_cost
FROM core.scored s;

-- ---- report 1: risk tier roll-up (the slide number) ----
CREATE OR REPLACE TABLE report.risk_tier_summary AS
SELECT f.risk_tier,
       d.action_guidance,
       COUNT(*)                                                        AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)              AS pct_base,
       ROUND(AVG(f.churn_probability), 4)                              AS mean_predicted,
       ROUND(AVG(f.churned), 4)                                        AS observed_churn_rate,
       ROUND(AVG(f.churned) / NULLIF(AVG(f.churn_probability), 0), 3)  AS calibration_ratio,
       ROUND(SUM(f."MonthlyCharges"), 0)                               AS monthly_revenue,
       ROUND(SUM(f.revenue_at_risk_12m), 0)                            AS revenue_at_risk_12m,
       ROUND(SUM(f.campaign_cost), 0)                                  AS campaign_cost,
       ROUND(SUM(f.revenue_at_risk_12m) * 0.30 - SUM(f.campaign_cost), 0) AS scenario_net_profit
FROM core.fact_churn_score f
JOIN core.dim_risk_tier d USING (risk_tier)
GROUP BY 1, 2
ORDER BY MIN(d.prob_low) DESC;

-- ---- report 2: segment churn, ordered by what actually hurts ----
CREATE OR REPLACE TABLE report.churn_by_segment AS
WITH seg AS (
    SELECT 'Contract' AS dimension, "Contract" AS segment, target AS churned, "MonthlyCharges" AS mrc
    FROM core.customer
    UNION ALL
    SELECT 'PaymentMethod', "PaymentMethod", target, "MonthlyCharges" FROM core.customer
    UNION ALL
    SELECT 'tenure_band', tenure_band, target, "MonthlyCharges" FROM core.customer_scored_features
    UNION ALL
    SELECT 'SeniorCitizen', CASE WHEN "SeniorCitizen" = 1 THEN 'senior' ELSE 'non-senior' END,
           target, "MonthlyCharges" FROM core.customer
    UNION ALL
    SELECT 'TechSupport', TechSupport, target, "MonthlyCharges" FROM core.customer
    UNION ALL
    SELECT 'OnlineSecurity', OnlineSecurity, target, "MonthlyCharges" FROM core.customer
    UNION ALL
    SELECT 'InternetService', InternetService, target, "MonthlyCharges" FROM core.customer
    UNION ALL
    SELECT 'PaperlessBilling', PaperlessBilling, target, "MonthlyCharges" FROM core.customer
)
SELECT dimension, segment,
       COUNT(*)                                       AS customers,
       ROUND(AVG(churned), 4)                         AS churn_rate,
       ROUND(AVG(churned) * 100 / (SELECT AVG(target * 100) FROM core.customer), 1)
                                                      AS index_vs_base,
       ROUND(AVG(mrc), 2)                             AS mean_monthly_charge,
       ROUND(SUM(churned * mrc), 0)                   AS lost_monthly_revenue
FROM seg
GROUP BY 1, 2
ORDER BY dimension, churn_rate DESC;

-- ---- report 3: cohort grid, tenure band x contract ----
CREATE OR REPLACE TABLE report.cohort_retention_grid AS
SELECT f.tenure_band,
       c."Contract",
       COUNT(*)                                   AS customers,
       ROUND(1 - AVG(c.target), 4)                AS retention_rate,
       ROUND(SUM(c.target), 0)                    AS churned,
       ROUND(AVG(s.churn_probability), 4)         AS mean_predicted_churn
FROM core.customer_scored_features f
JOIN core.customer c USING (customerID)
JOIN core.scored s USING (customerID)
GROUP BY 1, 2
ORDER BY 1, 2;

-- ---- report 4: the actual call list ----
-- Ordered by expected retained revenue, not raw probability: two customers at 0.92
-- are not equal if one pays $200/mo and the other $20.
CREATE OR REPLACE TABLE report.campaign_target_list AS
SELECT s.customerID,
       s.risk_tier,
       ROUND(s.churn_probability, 4)                          AS churn_probability,
       c."Contract",
       c."PaymentMethod",
       c."MonthlyCharges",
       c.tenure,
       f.service_count,
       f.no_support_flag,
       f.bill_shock_flag,
       ROUND(c."MonthlyCharges" * 12, 0)                      AS annual_revenue,
       ROUND(c."MonthlyCharges" * 12 * 0.30 - 40, 0)          AS expected_net_value,
       CASE WHEN c."Contract" = 'Month-to-month' AND f.electronic_check_flag = 1
            THEN 'Proactive support call + contract upgrade offer'
            WHEN f.no_support_flag = 1  THEN 'Proactive support call + tech-support bundle'
            WHEN f.bill_shock_flag = 1  THEN 'Discount / bill-review offer'
            WHEN c.tenure <= 12         THEN 'Product education + onboarding check-in'
            ELSE 'Loyalty reward offer' END                   AS recommended_action
FROM core.scored s
JOIN core.customer c USING (customerID)
JOIN core.customer_scored_features f USING (customerID)
WHERE s.risk_tier = 'High'
ORDER BY expected_net_value DESC;

-- ---- report 5: second-source enrichment (churn reasons, only in IBM 11.1.3+) ----
CREATE OR REPLACE TABLE report.churn_reason_by_tier AS
SELECT s.risk_tier,
       e."Churn Category",
       COUNT(*)                                  AS customers,
       ROUND(AVG(s.churn_probability), 4)        AS mean_predicted,
       ROUND(AVG(e."CLTV"), 0)                   AS mean_cltv
FROM core.scored s
JOIN ext.churn_reason e USING (customerID)
GROUP BY 1, 2
ORDER BY 1, customers DESC;

CREATE OR REPLACE TABLE report.top_churn_reasons AS
SELECT e."Churn Reason",
       e."Churn Category",
       COUNT(*)                              AS churned_customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM ext.churn_reason), 1) AS pct_of_churners,
       ROUND(AVG(e."CLTV"), 0)               AS mean_cltv,
       ROUND(AVG(e."Satisfaction Score"), 2) AS mean_satisfaction
FROM ext.churn_reason e
GROUP BY 1, 2
ORDER BY churned_customers DESC
LIMIT 15;

-- Geography only exists in the second source, so this report is second-source-only
-- and says so in its name rather than implying primary-feed coverage.
CREATE OR REPLACE TABLE report.churn_by_state_ext_source AS
SELECT l."State",
       COUNT(*)                                   AS customers,
       ROUND(AVG(c.target), 4)                    AS churn_rate,
       ROUND(AVG(s.churn_probability), 4)         AS mean_predicted,
       ROUND(AVG(e."CLTV"), 0)                    AS mean_cltv,
       ROUND(AVG(TRY_CAST(p."Population" AS BIGINT)), 0) AS mean_zip_population
FROM core.customer c
JOIN core.scored s USING (customerID)
JOIN ext.location l USING (customerID)
LEFT JOIN ext.customer_cltv e ON e.customerID = c.customerID
LEFT JOIN stg.zip_population p ON p."Zip Code" = l."Zip Code"
GROUP BY 1
ORDER BY customers DESC;

-- ---- report 6: model governance read-out (model card numbers in SQL) ----
CREATE OR REPLACE TABLE report.model_performance AS
SELECT m.model, m.roc_auc, m.pr_auc, m."f1_at_0.5" AS f1_at_0_5,
       m.top_decile_recall, m.top_decile_lift,
       CASE WHEN m.model = 'Logistic Regression' THEN 'deployed (isotonic-calibrated)'
            ELSE 'challenger' END AS status
FROM ext.model_comparison m
ORDER BY m.pr_auc DESC;

CREATE OR REPLACE TABLE report.fairness_by_segment AS
SELECT segment_dimension, segment, n, base_churn_rate, mean_predicted_prob, calibration_gap_pp,
       high_risk_share, recall_at_high_cutoff, false_positive_rate, precision_at_high_cutoff, roc_auc
FROM ext.fairness_by_segment
ORDER BY segment_dimension, n DESC;

CREATE OR REPLACE TABLE report.drift_watch AS
SELECT batch, measurement, value, verdict,
       CASE WHEN verdict IN ('WATCH', 'RETRAIN / INVESTIGATE', 'SCHEMA CHANGE')
            THEN 'action required' ELSE 'ok' END AS review_flag,
       detail
FROM ext.drift_report
WHERE measurement = 'score_psi'
   OR measurement LIKE 'category_change%'
   OR verdict IN ('WATCH', 'RETRAIN / INVESTIGATE', 'SCHEMA CHANGE')
   OR measurement LIKE 'realised%'
ORDER BY batch, verdict DESC, value DESC;

SELECT * FROM report.risk_tier_summary;
SELECT * FROM report.model_performance;
SELECT * FROM report.drift_watch;
SELECT COUNT(*) AS campaign_target_list_rows FROM report.campaign_target_list;
