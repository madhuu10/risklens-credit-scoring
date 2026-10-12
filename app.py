"""
RiskLens — Credit Default Risk Scoring
========================================
A Streamlit UI around a trained gradient-boosting model that predicts the
probability a loan applicant defaults, with per-prediction SHAP explanations,
batch scoring, and a model-performance page.

Run with:   streamlit run app.py
"""

import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="RiskLens", page_icon="💳", layout="wide")

# ---------------------------------------------------------------------
# Palette (validated categorical / status / diverging roles)
# ---------------------------------------------------------------------
BLUE = "#2a78d6"      # diverging pole: lowers risk / categorical slot 1
RED = "#e34948"       # diverging pole: raises risk
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
CAT8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

RISK_TIERS = [
    (0.15, "Low", GOOD),
    (0.45, "Medium", WARNING),
    (1.01, "High", CRITICAL),
]


def risk_tier(p):
    for cutoff, label, color in RISK_TIERS:
        if p < cutoff:
            return label, color
    return "High", CRITICAL


NUMERIC = ["age", "annual_income", "emp_length_years", "loan_amount",
           "interest_rate", "loan_percent_income", "credit_history_length", "previous_default"]
NOMINAL = ["home_ownership", "loan_purpose"]
ORDINAL = ["loan_grade"]
FEATURE_COLS = NUMERIC + NOMINAL + ORDINAL
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


# ---------------------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/artifacts/model.joblib")


@st.cache_resource
def load_explainer(_pipe):
    return shap.TreeExplainer(_pipe.named_steps["clf"])


@st.cache_data
def load_metrics():
    with open("model/artifacts/metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    return pd.read_csv("model/artifacts/feature_importance.csv")


@st.cache_data
def load_training_data():
    return pd.read_csv("data/credit_risk.csv")


@st.cache_data
def load_sample_applicants():
    return pd.read_csv("model/artifacts/sample_applicants.csv")


def clean_feature_name(raw):
    """'num__loan_percent_income' -> 'loan percent income'"""
    return raw.split("__", 1)[-1].replace("_", " ")


def predict_with_explanation(pipe, explainer, df):
    """Return (probabilities, shap_values, transformed_feature_names)."""
    proba = pipe.predict_proba(df[FEATURE_COLS])[:, 1]
    X_t = pipe.named_steps["prep"].transform(df[FEATURE_COLS])
    sv = explainer.shap_values(X_t)
    names = pipe.named_steps["prep"].get_feature_names_out()
    return proba, sv, names


pipe = load_model()
explainer = load_explainer(pipe)
metrics = load_metrics()
feat_importance = load_feature_importance()
train_df = load_training_data()

# ---------------------------------------------------------------------
# Global styling (mirrors the TradeRecon app's header/tier-card pattern)
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px;}}
    .app-header {{
        background: #0f172a; border-bottom: 3px solid {BLUE}; padding: 1.05rem 1.6rem;
        border-radius: 10px; display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 1.4rem;
    }}
    .app-header-left {{display: flex; align-items: center; gap: .85rem;}}
    .logo-mark {{
        width: 40px; height: 40px; border-radius: 9px; background: {BLUE}; color: #fff;
        display: flex; align-items: center; justify-content: center; font-weight: 800;
        font-size: 1.1rem; flex-shrink: 0;
    }}
    .app-title {{color: #fff; font-size: 1.12rem; font-weight: 700; line-height: 1.25;}}
    .app-subtitle {{color: #94a3b8; font-size: .8rem; margin-top: .1rem;}}
    .env-badge {{
        background: rgba(255,255,255,.08); color: #cbd5e1; font-size: .66rem; font-weight: 700;
        letter-spacing: .07em; padding: .32rem .65rem; border-radius: 20px;
        border: 1px solid rgba(255,255,255,.15); white-space: nowrap;
    }}
    .tier-badge {{
        display: inline-flex; align-items: center; gap: .5rem; padding: .55rem 1.1rem;
        border-radius: 10px; font-weight: 800; font-size: 1.3rem; color: #fff;
    }}
    .metric-card {{
        border-radius: 10px; padding: .9rem 1.05rem; background: #f8fafc;
        border: 1px solid #e2e8f0; height: 100%;
    }}
    .metric-card .label {{font-size: .74rem; color: {INK_SECONDARY}; text-transform: uppercase;
        letter-spacing: .04em; font-weight: 700;}}
    .metric-card .value {{font-size: 1.55rem; font-weight: 800; color: {INK}; margin-top: .2rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="logo-mark">RL</div>
            <div>
                <div class="app-title">RiskLens</div>
                <div class="app-subtitle">Credit Default Risk Scoring</div>
            </div>
        </div>
        <div class="env-badge">ML MODEL: {metrics['best_model'].upper().replace('_', ' ')} · AUC {metrics['comparison'][metrics['best_model']]['roc_auc']:.3f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_score, tab_batch, tab_perf, tab_about = st.tabs(
    ["Score an Applicant", "Batch Scoring", "Model Performance", "About the Data"]
)

# =======================================================================
# TAB 1 — Score a single applicant, with SHAP explanation
# =======================================================================
with tab_score:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("##### Applicant details")
        with st.form("score_form"):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 18, 100, 30)
            annual_income = c2.number_input("Annual income ($)", 1000, 2_000_000, 60000, step=1000)
            home_ownership = c1.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
            emp_length_years = c2.number_input("Employment length (years)", 0.0, 50.0, 4.0, step=0.5)
            loan_purpose = c1.selectbox(
                "Loan purpose",
                ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
            )
            loan_grade = c2.selectbox("Loan grade", GRADE_ORDER)
            loan_amount = c1.number_input("Loan amount ($)", 500, 50000, 10000, step=500)
            interest_rate = c2.number_input("Interest rate (%)", 1.0, 35.0, 11.0, step=0.1)
            credit_history_length = c1.number_input("Credit history length (years)", 0, 40, 5)
            previous_default = c2.checkbox("Prior default on file")
            submitted = st.form_submit_button("Score applicant", type="primary", width='stretch')

    if submitted:
        row = pd.DataFrame([{
            "age": age, "annual_income": annual_income, "home_ownership": home_ownership,
            "emp_length_years": emp_length_years, "loan_purpose": loan_purpose, "loan_grade": loan_grade,
            "loan_amount": loan_amount, "interest_rate": interest_rate,
            "loan_percent_income": round(loan_amount / annual_income, 4),
            "previous_default": int(previous_default), "credit_history_length": credit_history_length,
        }])
        proba, sv, names = predict_with_explanation(pipe, explainer, row)
        p = float(proba[0])
        tier, color = risk_tier(p)

        with right:
            st.markdown("##### Result")
            st.markdown(
                f"""<div style="display:flex; align-items:center; gap:1.2rem; margin-bottom:1rem;">
                    <div class="tier-badge" style="background:{color};">{tier} risk</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{INK};">{p:.1%}</div>
                    <div style="color:{INK_SECONDARY}; font-size:.85rem;">predicted probability of default</div>
                </div>""",
                unsafe_allow_html=True,
            )

            contrib = pd.DataFrame({
                "feature": [clean_feature_name(n) for n in names],
                "shap": sv[0],
            })
            contrib["abs"] = contrib["shap"].abs()
            top = contrib.sort_values("abs", ascending=False).head(8).sort_values("shap")
            top["direction"] = np.where(top["shap"] >= 0, "raises risk", "lowers risk")

            st.markdown("##### What drove this score")
            chart = (
                alt.Chart(top)
                .mark_bar(size=16, cornerRadius=3)
                .encode(
                    x=alt.X("shap:Q", title="Contribution to default probability (SHAP)"),
                    y=alt.Y("feature:N", sort=top["feature"].tolist(), title=None),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(domain=["lowers risk", "raises risk"], range=[BLUE, RED]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=["feature", alt.Tooltip("shap:Q", format=".3f"), "direction"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, width='stretch')
            st.caption(
                "Bars show each feature's push on this applicant's predicted default "
                "probability (SHAP values from the trained gradient-boosting model), "
                "not a causal claim."
            )
    else:
        with right:
            st.info("Fill in the applicant details and click **Score applicant**.")

# =======================================================================
# TAB 2 — Batch scoring
# =======================================================================
with tab_batch:
    st.markdown("##### Score a batch of applicants")
    st.caption(f"Upload a CSV with columns: {', '.join(FEATURE_COLS)}")

    c1, c2 = st.columns([2, 1])
    uploaded = c1.file_uploader("Applicants CSV", type="csv", label_visibility="collapsed")
    use_sample = c2.button("Use sample applicants", width='stretch')

    batch_df = None
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
    elif use_sample:
        batch_df = load_sample_applicants().drop(columns=["default_status"])

    if batch_df is not None:
        missing = set(FEATURE_COLS) - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        else:
            proba = pipe.predict_proba(batch_df[FEATURE_COLS])[:, 1]
            out = batch_df.copy()
            out["default_probability"] = proba.round(4)
            out["risk_tier"] = [risk_tier(p)[0] for p in proba]

            tier_counts = out["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            cols = st.columns(3)
            for col, (tier, color) in zip(cols, [("Low", GOOD), ("Medium", WARNING), ("High", CRITICAL)]):
                col.markdown(
                    f"""<div class="metric-card" style="border-left:4px solid {color};">
                        <div class="label">{tier} risk</div>
                        <div class="value">{int(tier_counts[tier]):,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            st.dataframe(out, width='stretch', height=400)
            st.download_button(
                "Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                "scored_applicants.csv", "text/csv",
            )
    else:
        st.info("Upload a CSV, or click **Use sample applicants** to try it with holdout data.")

# =======================================================================
# TAB 3 — Model performance
# =======================================================================
with tab_perf:
    st.markdown("##### Model comparison")
    comp = pd.DataFrame(metrics["comparison"]).T.reset_index().rename(columns={"index": "model"})
    comp_display = comp.copy()
    for c in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
        comp_display[c] = comp_display[c].round(3)
    st.dataframe(comp_display, width='stretch', hide_index=True)
    st.caption(f"Best model selected by test-set ROC-AUC: **{metrics['best_model']}**. "
               f"Trained on {metrics['n_train']:,} rows, evaluated on {metrics['n_test']:,} held-out rows.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ROC curve")
        roc_df = pd.DataFrame({"fpr": metrics["roc_curve"]["fpr"], "tpr": metrics["roc_curve"]["tpr"]})
        diag = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
        line = alt.Chart(roc_df).mark_line(color=BLUE, strokeWidth=2.5).encode(
            x=alt.X("fpr:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])),
        )
        ref = alt.Chart(diag).mark_line(color=MUTED, strokeDash=[4, 4]).encode(x="fpr:Q", y="tpr:Q")
        st.altair_chart((line + ref).properties(height=300), width='stretch')

    with c2:
        st.markdown("##### Confusion matrix (test set, 0.5 cutoff)")
        cm = np.array(metrics["confusion_matrix"]["matrix"])
        labels = metrics["confusion_matrix"]["labels"]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt(
            id_vars="index", var_name="predicted", value_name="count"
        ).rename(columns={"index": "actual"})
        heat = (
            alt.Chart(cm_df)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", sort=labels),
                y=alt.Y("actual:N", sort=labels),
                color=alt.Color("count:Q", scale=alt.Scale(range=SEQ_BLUE), legend=None),
                tooltip=["actual", "predicted", "count"],
            )
            .properties(height=300)
        )
        text = heat.mark_text(baseline="middle", fontWeight="bold").encode(
            text="count:Q", color=alt.condition(alt.datum.count > cm.max() / 2, alt.value("white"), alt.value(INK))
        )
        st.altair_chart(heat + text, width='stretch')

    st.markdown("##### Global feature importance")
    top_feat = feat_importance.head(10).copy()
    top_feat["feature"] = top_feat["feature"].apply(clean_feature_name)
    bar = (
        alt.Chart(top_feat)
        .mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(bar, width='stretch')

# =======================================================================
# TAB 4 — About the data
# =======================================================================
with tab_about:
    st.markdown("##### Dataset")
    st.markdown(
        "31,415 historical loan applications with a binary default outcome, sourced from "
        "[`dalim09/credit-risk-eda`](https://huggingface.co/datasets/dalim09/credit-risk-eda) "
        "on the Hugging Face Hub. Overall default rate: "
        f"**{metrics['default_rate']:.1%}**."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### Default rate by loan grade")
        grade_rate = train_df.groupby("loan_grade")["default_status"].mean().reindex(GRADE_ORDER).reset_index()
        chart = (
            alt.Chart(grade_rate)
            .mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("loan_grade:N", title="Loan grade", sort=GRADE_ORDER),
                y=alt.Y("default_status:Q", title="Default rate", axis=alt.Axis(format="%")),
                tooltip=["loan_grade", alt.Tooltip("default_status:Q", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width='stretch')

    with c2:
        st.markdown("###### Default rate by home ownership")
        home_rate = train_df.groupby("home_ownership")["default_status"].mean().reset_index()
        chart = (
            alt.Chart(home_rate)
            .mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("home_ownership:N", title=None, sort="-y"),
                y=alt.Y("default_status:Q", title="Default rate", axis=alt.Axis(format="%")),
                tooltip=["home_ownership", alt.Tooltip("default_status:Q", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width='stretch')

    st.markdown("###### Loan purpose distribution")
    purpose_counts = train_df["loan_purpose"].value_counts().reset_index()
    purpose_counts.columns = ["loan_purpose", "count"]
    chart = (
        alt.Chart(purpose_counts)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("count:Q", title="Applications"),
            y=alt.Y("loan_purpose:N", sort="-x", title=None),
            color=alt.Color("loan_purpose:N", scale=alt.Scale(range=CAT8), legend=None),
            tooltip=["loan_purpose", "count"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width='stretch')

    st.caption(
        "Nothing on this page is financial advice; this is a portfolio demonstration of a "
        "default-risk scoring model, not a production underwriting system."
    )
