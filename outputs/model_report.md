# Model report

## Comparison (test split, n=1761)

| model                     |   roc_auc |   pr_auc |   f1_at_0.5 |   top_decile_recall |   top_decile_lift |
|:--------------------------|----------:|---------:|------------:|--------------------:|------------------:|
| Logistic Regression       |    0.8493 |   0.6625 |      0.6234 |              0.2848 |              2.85 |
| LightGBM (Optuna-tuned)   |    0.8472 |   0.6538 |      0.6341 |              0.2741 |              2.74 |
| Feedforward NN (PyTorch)  |    0.8468 |   0.6521 |      0.622  |              0.2848 |              2.85 |
| XGBoost                   |    0.8365 |   0.6363 |      0.5659 |              0.2741 |              2.74 |
| LightGBM                  |    0.8334 |   0.6354 |      0.627  |              0.2805 |              2.81 |
| Random Forest             |    0.8282 |   0.62   |      0.5401 |              0.2634 |              2.63 |
| MLP (sklearn feedforward) |    0.7632 |   0.5068 |      0.5248 |              0.2163 |              2.16 |

Best by PR-AUC: **Logistic Regression**. Deployed artifact: **model_calibrated.pkl** (isotonic, 3-fold) - scoring, batch and API all use it.

## Calibration

- ROC-AUC 0.8493 -> 0.8495 after calibration (isotonic is rank-preserving per fold, so the change is the 3-fold averaging, not a ranking win).
- Brier 0.16210 -> 0.13363; constant-base-rate baseline 0.19486.
- Share flagged High risk on test at cutoff 0.6: 13.3% (base churn 26.5%).

## Threshold and profit

- Profit-maximising threshold (scenario): 0.15000000000000002 -> $74,571 on test.
- Best contact fraction by scenario profit: 50% ($74,611).
- Assumptions: save rate 30%, $40 per contacted customer, CLV = monthly x 12. Scenario only - true uplift needs A/B test data.

## Fairness (see outputs/fairness_by_segment.csv)

| segment_dimension   |   groups |   max_high_risk_share |   min_high_risk_share |   selection_ratio_min_over_max |   recall_spread_pp |   recall_ratio |   max_abs_calibration_gap_pp | reference_group   | adverse_impact_note                                                                                                                                                                              |
|:--------------------|---------:|----------------------:|----------------------:|-------------------------------:|-------------------:|---------------:|-----------------------------:|:------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Contract            |        3 |                0.2417 |                0      |                          0     |              40.58 |          0     |                         0.82 | Month-to-month    | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |
| Dependents          |        2 |                0.1787 |                0.0296 |                          0.166 |              25.85 |          0.366 |                         1.75 | No                | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |
| Partner             |        2 |                0.2018 |                0.0621 |                          0.308 |              19.65 |          0.543 |                         0.48 | No                | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |
| SeniorCitizen       |        2 |                0.2727 |                0.1058 |                          0.388 |              13.44 |          0.707 |                         0.42 | senior citizen    | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |
| age_band            |        4 |                0.2727 |                0.0902 |                          0.331 |              15.9  |          0.654 |                         2.72 | 65+               | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |
| gender              |        2 |                0.1392 |                0.1263 |                          0.907 |               5.44 |          0.86  |                         0.93 | Male              | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |
| tenure_band         |        4 |                0.3708 |                0      |                          0     |              58.17 |          0     |                         2.46 | 0-12              | being flagged here means receiving a retention offer, so a higher selection rate is not by itself a harm; equal-opportunity (recall) and calibration gaps are the fairness questions that matter |

## Survival (lifelines + R, run independently)

- Median survival by contract: Month-to-month = 35.0 months, One year = not reached, Two year = not reached.
- R (`r/survival_analysis.R`, survival 3.8.6) and lifelines agree on the Cox coefficients to 6 decimal places; the side-by-side table is below and the raw R output is in outputs/r/.

| covariate             |   coef_lifelines |    coef_r |   hazard_ratio_lifelines |   hazard_ratio_r |   max_abs_coef_diff |
|:----------------------|-----------------:|----------:|-------------------------:|-----------------:|--------------------:|
| MonthlyCharges        |         0.024864 |  0.024864 |                 1.02517  |         1.02517  |                   0 |
| SeniorCitizen         |        -0.105391 | -0.105391 |                 0.899973 |         0.899973 |                   0 |
| service_count         |        -0.516359 | -0.516359 |                 0.596689 |         0.596689 |                   0 |
| contract_risk         |         1.71424  |  1.71424  |                 5.55244  |         5.55244  |                   0 |
| electronic_check_flag |         0.495433 |  0.495433 |                 1.64121  |         1.64121  |                   0 |
| no_support_flag       |         0.040913 |  0.040913 |                 1.04176  |         1.04176  |                   0 |
| bill_shock_flag       |         0.82999  |  0.82999  |                 2.2933   |         2.2933   |                   0 |

- **Proportional-hazards assumption fails.** R `cox.zph` (identity time transform) rejects it for MonthlyCharges, service_count, contract_risk and no_support_flag; lifelines' rank test rejects the same three drivers. The two implementations disagree on the two weakest terms (no_support_flag, bill_shock_flag) because they transform time differently.
- Consequence: the hazard ratios above are **time-averaged** effects, valid for direction and magnitude over the observed window, not as instantaneous effects at a given tenure. The KM curves and the classification model carry the operational conclusions; the Cox model is supporting evidence. A fix (stratify on contract_risk, or add time-varying coefficients) is out of scope here and is listed in the README limitations.

| term                  |     chisq | p_value   |   ph_test_p_rank |
|:----------------------|----------:|:----------|-----------------:|
| MonthlyCharges        |  94.411   | < 1e-16   |      0.00246063  |
| SeniorCitizen         |   0.40342 | 0.5253    |      0.427984    |
| service_count         | 163.326   | < 1e-16   |      1.74826e-08 |
| contract_risk         |  76.2197  | < 1e-16   |      2.32818e-14 |
| electronic_check_flag |   0.70164 | 0.4022    |      0.882699    |
| no_support_flag       |  10.9517  | 0.0009352 |      0.286188    |
| bill_shock_flag       |   1.78637 | 0.1814    |      0.0172419   |
| GLOBAL                | 248.563   | < 1e-16   |    nan           |

## Drift (full table: outputs/drift_report.csv)

- **ibm_q3_rerelease** (7043 rows, churn 26.5%) - score PSI **0.0201** (STABLE); 1 of 32 features above PSI 0.10; 1 categorical domain change(s); realised ROC-AUC 0.8645, PR-AUC 0.7022, precision at the 0.6 cutoff 0.807.
  IBM 11.1.3 Q3 re-release mapped to the 21-column schema. Same 7,043 customers and same labels as the reference; categories re-coded (Electronic check removed, DSL split into DSL/Cable, 265 contracts moved to term). Measures upstream re-coding drift, NOT temporal drift.
- **held_out_test_split** (1761 rows, churn 26.5%) - score PSI **0.0038** (STABLE); 0 of 32 features above PSI 0.10; 0 categorical domain change(s); realised ROC-AUC 0.8495, PR-AUC 0.6637, precision at the 0.6 cutoff 0.718.
  Stratified test split, never used for training or calibration. Same period as the reference by construction, so a near-zero PSI here is a control, not evidence of stability over time.
