# EDA insight report - Customer Churn (IBM Telco, n=7,043)

- Data quality score: 100.0% (35/35 checks passed, 2 of them Pandera schema validations over all 21 columns). Blank TotalCharges: 11, all tenure-0.
- Base churn rate is 26.5% (1869 of 7043 customers).
- Highest churn in Contract: Month-to-month at 42.7% (n=3875), vs base 26.5%.
- Highest churn in PaymentMethod: Electronic check at 45.3% (n=2365), vs base 26.5%.
- Highest churn in InternetService: Fiber optic at 41.9% (n=3096), vs base 26.5%.
- Highest churn in TechSupport: No at 41.6% (n=3473), vs base 26.5%.
- Highest churn in OnlineSecurity: No at 41.8% (n=3498), vs base 26.5%.
- Highest churn in PaperlessBilling: Yes at 33.6% (n=4171), vs base 26.5%.
- Highest churn in SeniorCitizen: 1.0 at 41.7% (n=1142), vs base 26.5%.
- Highest churn in Partner: No at 33.0% (n=3641), vs base 26.5%.
- Highest churn in Dependents: No at 31.3% (n=4933), vs base 26.5%.
- Churn by tenure band: 0-12 47.4% (n=2186), 13-24 28.7% (n=1024), 25-48 20.4% (n=1594), 49-72 9.5% (n=2239)
- Month-to-month customers: 3875, churn 42.7% vs 26.5% base.
- Month-to-month + Fiber optic + Electronic check: 1307 customers, churn 60.4% - 2.3x the base rate.
- Month-to-month + no TechSupport: 2680 customers, churn 50.4% - 1.9x the base rate.
- PyTorch feedforward network (64-32, BatchNorm + 0.3 dropout, Adam, class-weighted BCE, early-stopped at epoch 27) reached ROC-AUC 0.847 vs 0.763 for the sklearn MLP.
- Best-by-PR-AUC model Logistic Regression was isotonic-calibrated over 3 folds; scoring and tiering use outputs/model_calibrated.pkl, not the raw scores.
- Tree SHAP top driver for the Optuna-tuned LightGBM: cat__Contract_Month-to-month (mean |SHAP| 0.636).
- Fairness: seniors are 16.2% of the test split, base churn 42.7% vs non-seniors 23.4%; flagged as High risk 27.3% vs 10.6%, recall at the cutoff 45.9% vs 32.5%.
- IBM's own supplied Churn Score reaches ROC-AUC 0.945 / PR-AUC 0.878 on the same 1,761 held-out customers, against 0.850 / 0.664 for our best calibrated model. We do not close that gap and we do not pretend to: nothing built from the 21 published columns clears 0.85 ROC-AUC here (seven model families tried, including a tuned gradient-boosting stack and a PyTorch network), so Churn Score is computed from inputs that are not in this feed and IBM does not document. It is excluded as a feature - it encodes the answer - and reported here as an external yardstick. Spearman between the two scores is only 0.36, so they are not measuring the same ordering of customers.
- Drift on the IBM Q3 re-release batch: score PSI 0.0201 (STABLE) - the model output barely moved, but the feed did: PaymentMethod PSI 3.3899 (RETRAIN / INVESTIGATE). Realised ROC-AUC on that batch 0.8645 vs 0.8495 on the reference test split - the re-coded payment field erases the Electronic check signal upstream.

## What is NOT in this dataset (do not claim it)
- No support-call log, login log, usage 30/60/90 series, complaint log, payment-delay or signup-date field.
- Bill shock, contract-remaining and adapted RFM are proxies from available columns only.
- A true time-based split needs dated snapshots; this build uses a stratified split.
- The second source (IBM 11.1.3) is a re-coded re-release of the same book of business, not a later period - see data/raw/ibm_telco_11_1_3/PROVENANCE.md.
