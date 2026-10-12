"""
One-off data pull: Hugging Face Hub -> cleaned local CSV.

Run once (`python fetch_data.py`) to populate data/credit_risk.csv. The rest
of the project (training, the app) reads only that local file — no network
dependency at train or serve time.

Source: https://huggingface.co/datasets/dalim09/credit-risk-eda
31,415 loan applications with a binary default label. Column names on the
Hub use display case and the "Age Group" column has a mangled en-dash from
a source encoding issue; both are fixed here.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

RENAME = {
    "Age": "age",
    "Annual Income": "annual_income",
    "Home Ownership": "home_ownership",
    "Employment Length (Years)": "emp_length_years",
    "Loan Purpose": "loan_purpose",
    "Loan Grade": "loan_grade",
    "Loan Amount": "loan_amount",
    "Interest Rate": "interest_rate",
    "Default Status": "default_status",
    "Loan % of Income": "loan_percent_income",
    "Previous Default": "previous_default",
    "Credit History Length (Years)": "credit_history_length",
}

# Engineered bucket columns shipped with the source file are dropped — they're
# just re-binned versions of columns already present, not new information, and
# the app builds its own display buckets where needed.
DROP = ["LTI Group", "Income Group", "Age Group", "Interest Rate Group"]


def main():
    path = hf_hub_download("dalim09/credit-risk-eda", "data.csv", repo_type="dataset")
    df = pd.read_csv(path)
    df = df.drop(columns=DROP).rename(columns=RENAME)
    df["previous_default"] = (df["previous_default"] == "Yes").astype(int)
    df.to_csv("data/credit_risk.csv", index=False)
    print(f"Wrote data/credit_risk.csv — {len(df):,} rows, {len(df.columns)} columns")
    print(f"Default rate: {df['default_status'].mean():.2%}")


if __name__ == "__main__":
    main()
