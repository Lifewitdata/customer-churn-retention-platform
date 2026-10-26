# Second data source — IBM "Telco customer churn (11.1.3+)"

The primary feed for this project is `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
(the classic 7,043 x 21 IBM sample). The six `.xlsx` modules here are IBM's later
11.1.3 release of the same book of business, split into modules with fields the
classic CSV does not carry.

| Module | Rows | What it adds |
| --- | --- | --- |
| `Telco_customer_churn.xlsx` | 7,043 | Country/State/City/Zip/Lat-Long, `CLTV`, `Churn Score`, `Churn Reason` |
| `Telco_customer_churn_demographics.xlsx` | 7,043 | `Age`, `Under 30`, `Married`, `Number of Dependents` |
| `Telco_customer_churn_services.xlsx` | 7,043 | `Quarter`, `Offer`, referrals, internet type (incl. Cable), avg GB, long-distance / extra-data charges, `Total Revenue` |
| `Telco_customer_churn_status.xlsx` | 7,043 | `Satisfaction Score`, `Customer Status` (Joined/Stayed/Churned), `Churn Category`, `Churn Reason` |
| `Telco_customer_churn_location.xlsx` | 7,043 | geography keyed by customer |
| `Telco_customer_churn_population.xlsx` | 1,671 | Zip-level population (US Census 2010) |

## Provenance

- Announced by Steven Macko (IBM) in *Telco customer churn (11.1.3+)*,
  2019-07-11: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019-07-11/telco-customer-churn-1113
- Also mirrored on Kaggle as `ylchang/telco-customer-churn-1113`:
  https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113 (login-walled, so not used for download)
- Downloaded for this build from a public GitHub mirror of the IBM modules:
  https://github.com/Dina-Hosny/Telco-Customer-Churn-Analysis-using-Power-BI/tree/main/DataSet
- Fetch date: 2026-10-26. These are third-party copies of an IBM sample dataset; checksums below pin
  exactly what this build was run against.

```
1bcbc0ccc9b352175216979102628e579cfbde2c3ff57b005de168a433122640  Telco_customer_churn.xlsx
2e4062dc0d77bc6567634b8697ff3a18466c502e489bed2c8b7b97b7588d3f34  Telco_customer_churn_demographics.xlsx
7bac60961b08cf907e83016346978224f9a678a7c317df7d6dda90bcb84e57bb  Telco_customer_churn_location.xlsx
dc410f5f9402a699013b93d8efae727e7bd91e25a0d56a05801e4376932e7740  Telco_customer_churn_population.xlsx
518cb73070028d84c741dc91d2dbf55f629d8a856c119cedfee36a97a2f96c92  Telco_customer_churn_services.xlsx
fc256ac6e2ea4e5036a28401a5e4ae786ad4b14893dfb0a3a4817cc8c3ec9289  Telco_customer_churn_status.xlsx
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91  ../WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## What this second source is used for

1. **Enrichment / analysis-only fields** — churn reason + category breakdown of the
   actual churners, CLTV-based value at risk, geography and satisfaction segments.
   Written to `outputs/enrichment_*.csv` by the pipeline.
2. **Cross-source reconciliation** — `outputs/source_reconciliation.csv` compares key
   coverage and field agreement between the two feeds.
3. **A re-release / upstream-schema-change drift batch** — `src/external.py` maps the
   Q3-tagged services + demographics + status modules back into the 21-column schema so
   the shipped model can score them, and `src/monitor.py` measures score PSI, feature
   PSI and realised discrimination on it.

## What this second source is explicitly NOT

It is **not a later time period, and not a second population**. Verified in this build
(`outputs/source_reconciliation.csv`): tenure, monthly charges and churn agree with the
primary CSV for 99.8-100% of the same 7,043 customer IDs. IBM re-coded some categories:

- `PaymentMethod` in the Q3 module has only three values (Bank Withdrawal, Credit Card,
  Mailed Check) — the 2,365 **Electronic check** customers are re-labelled
  *Bank Withdrawal*, so the strongest payment signal in the model is erased upstream.
- `Internet Type` splits DSL into **DSL** and **Cable** (769 rows); Cable is mapped to DSL here.
- Contract labels differ only in title-casing (`Month-to-Month` vs `Month-to-month`).

So drift measured on this batch is **re-release / re-coding drift**, which is a real and
common production failure mode (an upstream feed changes its encoding), but it must not be
described as temporal drift. Any claim about time-based drift needs a genuinely dated
later snapshot, which this dataset family does not provide.

## Leakage note

`Churn Reason`, `Churn Category`, `Churn Score`, `CLTV` and `Customer Status` are only
known for (or about) customers whose outcome is already observed. They are used for
post-hoc description of churners and are **never** model features. The feature list in
`src/features.py` is unchanged by this second source.
