# FastAPI endpoint — live smoke test

Item 15 of the review asked for the endpoint to actually be run, not just imported. It
had never worked: `POST /score` returned **500 `KeyError: 'tenure_band'`**, because
`_score_frame()` coerced every entry of `features.CAT`, and `tenure_band` is a column
`engineer()` *derives* — no caller supplies it. The fix is `features.CAT_RAW`
(the 15 categorics a caller must send, i.e. `CAT` minus `tenure_band`), and the endpoint
now validates categories with `Literal[...]` instead of `examples=[...]`.

## Environment

| | |
|---|---|
| Python | 3.11.16 (project `.venv`) |
| fastapi | 0.162.0 |
| starlette | 1.10.0 |
| uvicorn | 0.54.0 |
| pydantic | 2.14.1 |
| model served | `outputs/model_calibrated.pkl` — the isotonic-calibrated best model (base estimator named in `outputs/evaluation.json`) |

## Reproduce

```bash
# terminal 1 — from the project root
.venv/Scripts/python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8123

# terminal 2
curl -sS http://127.0.0.1:8123/health
curl -sS -X POST http://127.0.0.1:8123/score \
  -H 'Content-Type: application/json' \
  -d '{"customerID":"5178-LMXOP","gender":"Male","SeniorCitizen":1,"Partner":"Yes",
       "Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"Yes",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes",
       "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":95.1,"TotalCharges":95.1}'
```

## Transcript (2026-10-26, captured over real HTTP, not `TestClient`)

```
GET /health
{"status":"ok","model":"model_calibrated.pkl","model_present":true,"high_risk_cutoff":0.6,"medium_risk_cutoff":0.35,"n_features":32}
HTTP 200  time=0.003129s

POST /score   5178-LMXOP — held-out churner, committed churn_probability 1.0
{"customerID":"5178-LMXOP","churn_probability":1.0,"risk_tier":"High","recommended_action":"Proactive support call + contract upgrade offer","revenue_at_risk_12m":1141.2,"model":"model_calibrated.pkl"}
HTTP 200

POST /score   1862-QRWPE — committed churn_probability 0.0
{"customerID":"1862-QRWPE","churn_probability":0.0,"risk_tier":"Low","recommended_action":"Monitor / standard care","revenue_at_risk_12m":0.0,"model":"model_calibrated.pkl"}
HTTP 200

POST /score/batch   2 customers
{"n":2,"counts_by_tier":{"High":1,"Low":1},"customers":[{"customerID":"7590-VHVEG","churn_probability":1.0,"risk_tier":"High","recommended_action":"Proactive support call + contract upgrade offer"},{"customerID":"1862-QRWPE","churn_probability":0.0,"risk_tier":"Low","recommended_action":"Monitor / standard care"}]}
HTTP 200

POST /score   out-of-domain PaymentMethod "Email check", Contract "Monthly", MonthlyCharges -5
{"detail":[{"type":"literal_error","loc":["body","Contract"],"msg":"Input should be 'Month-to-month', 'One year' or 'Two year'","input":"Monthly"},
{"type":"literal_error","loc":["body","PaymentMethod"],"msg":"Input should be 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check' or 'Mailed check'","input":"Email check"},
{"type":"greater_than","loc":["body","MonthlyCharges"],"msg":"Input should be greater than 0","input":-5}]}
HTTP 422

POST /score/batch   []
{"detail":"send between 1 and 5000 customers"}
HTTP 422

GET /docs          HTTP 200
GET /openapi.json  paths = ['/health', '/score', '/score/batch']
```

Server-side access log written by uvicorn for the same run
(`outputs/logs/uvicorn.log`, original file):

```
INFO:     Started server process [20516]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8123 (Press CTRL+C to quit)
INFO:     127.0.0.1:50908 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:50919 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:50920 - "POST /score HTTP/1.1" 200 OK
INFO:     127.0.0.1:50921 - "POST /score HTTP/1.1" 200 OK
INFO:     127.0.0.1:50922 - "POST /score/batch HTTP/1.1" 200 OK
INFO:     127.0.0.1:50923 - "POST /score HTTP/1.1" 422 Unprocessable Entity
INFO:     127.0.0.1:50924 - "POST /score/batch HTTP/1.1" 422 Unprocessable Entity
INFO:     127.0.0.1:50925 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:50926 - "GET /openapi.json HTTP/1.1" 200 OK
```

## Why the two scores above are the interesting part

`5178-LMXOP` and `1862-QRWPE` are not invented payloads. They are rows from
`outputs/scored_customers.csv`, so the probability the wire returns can be compared with
the probability the *pipeline* recorded for the same customer. They match exactly, which
is the claim that matters: the endpoint serves `model_calibrated.pkl`, the same artifact
`score_batch.py` and the dashboard load, so an API response and a nightly batch row for
one customer cannot disagree.

`revenue_at_risk_12m` = 1141.2 is `MonthlyCharges 95.10 × 12`, and it is 0.0 for the Low
tier — the field is only populated above the 0.60 High cutoff.

The 422 case is the guard that `OneHotEncoder(handle_unknown="ignore")` makes necessary.
Without domain validation, `"Email check"` (a category that exists in the Q3 re-release
but not the reference) encodes to an all-zero row and returns a confident, meaningless
probability. A typo is rejected at the boundary instead of becoming a business decision.

## Automated coverage

`tests/test_api.py` runs 23 assertions over `TestClient`, including a 25-customer parity
sweep against `outputs/scored_customers.csv` (`abs=1e-4`), batch-vs-single equality, and
domain/range rejection for each categorical and numeric field.

```bash
.venv/Scripts/python.exe -m pytest tests/test_api.py -q
# 23 passed
```
