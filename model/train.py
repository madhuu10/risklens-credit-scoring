"""
Train and compare credit-default models, save the winning pipeline + metrics.

Run from the project root: `python model/train.py`

Compares a logistic-regression baseline against random forest and gradient
boosting classifiers (all wrapped in the same preprocessing pipeline), picks
the best by test-set ROC-AUC, and persists:
  - model/artifacts/model.joblib        the winning sklearn Pipeline (preprocessing + classifier)
  - model/artifacts/metrics.json        comparison table + confusion matrix + ROC curve points
  - model/artifacts/feature_importance.csv
  - model/artifacts/sample_applicants.csv   holdout rows for the app's batch-scoring demo
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = [
    "age", "annual_income", "emp_length_years", "loan_amount",
    "interest_rate", "loan_percent_income", "credit_history_length", "previous_default",
]
NOMINAL = ["home_ownership", "loan_purpose"]
ORDINAL = ["loan_grade"]
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]  # best -> worst credit grade
TARGET = "default_status"

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42,
    ),
}


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
            ("ord", OrdinalEncoder(categories=[GRADE_ORDER]), ORDINAL),
        ]
    )


def evaluate(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def main():
    df = pd.read_csv("data/credit_risk.csv")
    X = df[NUMERIC + NOMINAL + ORDINAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    fitted = {}
    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe
        print(name, results[name])

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name}")

    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_points = list(zip(fpr[::20].tolist() + [1.0], tpr[::20].tolist() + [1.0]))

    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        importance = np.abs(clf.coef_[0])
    pd.DataFrame({"feature": feature_names, "importance": importance}) \
        .sort_values("importance", ascending=False) \
        .to_csv("model/artifacts/feature_importance.csv", index=False)

    metrics = {
        "best_model": best_name,
        "comparison": results,
        "confusion_matrix": {"labels": ["no_default", "default"], "matrix": cm},
        "roc_curve": {"fpr": [p[0] for p in roc_points], "tpr": [p[1] for p in roc_points]},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate": float(y.mean()),
    }
    with open("model/artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(best_pipe, "model/artifacts/model.joblib")

    # Holdout sample (with true labels) for the app's "try batch scoring" demo.
    sample = X_test.copy()
    sample[TARGET] = y_test.values
    sample.sample(n=min(300, len(sample)), random_state=42).to_csv(
        "model/artifacts/sample_applicants.csv", index=False
    )

    print(f"\nSaved model.joblib, metrics.json, feature_importance.csv, sample_applicants.csv")


if __name__ == "__main__":
    main()
