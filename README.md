# RiskLens — Credit Default Risk Scoring

A machine-learning web app that scores loan applicants by probability of
default, with per-prediction explanations. Built on 31,415 real loan
applications from the Hugging Face Hub.

**Live demo:** _add your deployed Streamlit Community Cloud URL here_

---

## What it does

- **Score an applicant** — enter an applicant's details and get a calibrated
  default probability, a Low / Medium / High risk tier, and a chart of the
  top factors that pushed the score up or down for *that specific applicant*
  (via [SHAP](https://github.com/shap/shap) on the trained model).
- **Batch scoring** — upload a CSV of applicants and download them back with
  a predicted default probability and risk tier per row.
- **Model performance** — compares three trained models (logistic
  regression, random forest, gradient boosting) on ROC-AUC, precision,
  recall, and F1; shows the winning model's ROC curve, confusion matrix, and
  global feature importance.
- **About the data** — default-rate breakdowns by loan grade, home
  ownership, and loan purpose.

## Why it exists

A reconciliation-style exercise (matching bills to payments) is useful but
rule-based — it doesn't involve training or evaluating a model. This project
is the ML-first complement: a supervised classification problem, model
comparison, and an explainability layer (SHAP) on top, wrapped in an app
someone other than me could actually use.

## The model

- **Data**: [`dalim09/credit-risk-eda`](https://huggingface.co/datasets/dalim09/credit-risk-eda)
  on the Hugging Face Hub — 31,415 loan applications, 21.6% default rate.
  Pulled once via `fetch_data.py` into `data/credit_risk.csv`; nothing at
  train or serve time touches the network.
- **Features**: age, annual income, home ownership, employment length, loan
  purpose, loan grade, loan amount, interest rate, loan-to-income ratio,
  prior default on file, credit history length.
- **Models compared** (`model/train.py`): logistic regression (baseline),
  random forest, gradient boosting — all behind the same
  `ColumnTransformer` preprocessing (standard-scaled numerics, one-hot
  nominal categories, ordinal-encoded loan grade A→G).
- **Winner**: gradient boosting, selected by test-set ROC-AUC.

  | Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
  |---|---|---|---|---|---|
  | Logistic regression | 0.865 | 0.794 | 0.515 | 0.773 | 0.618 |
  | Random forest | 0.931 | 0.918 | 0.863 | 0.737 | 0.795 |
  | **Gradient boosting** | **0.936** | **0.932** | **0.960** | 0.712 | **0.818** |

- **Explainability**: `shap.TreeExplainer` computes per-prediction feature
  contributions on the live model — not a static importance chart, an
  explanation specific to the applicant just scored.

## Known limitations

- This is a **portfolio demonstration**, not a production underwriting
  model — it has not been validated for fairness/bias across protected
  classes, calibrated against a real lender's loss data, or reviewed for
  regulatory compliance (e.g. adverse-action-notice requirements). Don't use
  it to make real credit decisions.
- The Low/Medium/High risk-tier cutoffs (15% / 45%) are illustrative
  thresholds, not derived from a cost-of-capital or expected-loss analysis.
- The source dataset's categorical encodings (loan grade, purpose) reflect
  whatever scheme the original lender used — they're taken as given, not
  independently verified.

## Project layout

```
fetch_data.py                  one-off: Hugging Face Hub -> data/credit_risk.csv
data/credit_risk.csv           cleaned dataset (committed, so train/serve need no network)
model/train.py                 trains & compares models, saves the winner + metrics
model/artifacts/
  model.joblib                 the winning sklearn Pipeline (preprocessing + classifier)
  metrics.json                 model comparison, confusion matrix, ROC curve points
  feature_importance.csv       global feature importance from the winning model
  sample_applicants.csv        holdout rows for the app's "try batch scoring" demo
app.py                         the Streamlit app
requirements.txt               pinned dependencies
.streamlit/config.toml         theme
```

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. The trained model and data are already
committed, so this works without re-running `fetch_data.py` or
`model/train.py` — those are only needed if you want to retrain.

To retrain from scratch:

```bash
python fetch_data.py
python model/train.py
```

## Deploying

1. Push this repo to GitHub (public, for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick this repo/branch, set the main file to
   `app.py`, and deploy.
3. Paste the resulting URL into this README's **Live demo** line above.
