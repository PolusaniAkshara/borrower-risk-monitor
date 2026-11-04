import gzip
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date
from sklearn.utils.validation import check_is_fitted


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Borrower Risk Predictor",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    :root {
        --navy: #10243A;
        --navy-light: #1B3654;
        --teal: #0F9D91;
        --teal-dark: #087F78;
        --ink: #172B46;
        --muted: #61738A;
        --line: #E1E9F2;
        --surface: #FFFFFF;
    }
    .stApp { background: linear-gradient(180deg, #F2F7FC 0%, #F8FAFD 55%, #F3F7FB 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10243A 0%, #183753 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #F4F8FC; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.08);
        border-radius: 12px; padding: 11px 13px; margin: 6px 0; width: 100%;
        transition: all .2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(15,157,145,.22); border-color: rgba(91,220,207,.35);
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    p, li { color: #34465D; }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1440px; }
    .brand-hero {
        position: relative; overflow: hidden; padding: 25px 30px 23px;
        border-radius: 22px; margin: 0 0 25px 0;
        background: linear-gradient(115deg, #10243A 0%, #1B4562 64%, #087F78 120%);
        box-shadow: 0 12px 30px rgba(16,36,58,.14);
        border: 1px solid rgba(255,255,255,.14);
    }
    .brand-hero:after {
        content: ""; position: absolute; width: 230px; height: 230px; right: -55px; top: -115px;
        border: 34px solid rgba(255,255,255,.07); border-radius: 50%; pointer-events: none;
    }
    .brand-title { color: #FFFFFF !important; font-size: clamp(1.7rem, 3vw, 2.35rem); font-weight: 800; line-height: 1.15; margin: 0; }
    .brand-subtitle { color: #D5E7F2 !important; font-size: .98rem; margin: 9px 0 0 0; max-width: 760px; }
    .brand-pill { display: inline-block; background: rgba(15,157,145,.22); color: #BDF7EF;
        border: 1px solid rgba(141,245,230,.3); padding: 5px 10px; border-radius: 999px;
        font-size: .75rem; font-weight: 700; letter-spacing: .04em; margin-bottom: 11px; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.96); border: 1px solid var(--line);
        padding: 17px 18px; border-radius: 16px; box-shadow: 0 5px 16px rgba(31,55,80,.045);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 11px; font-weight: 700; padding: .58rem 1rem;
        border: 1px solid #C9D8E5; transition: all .18s ease;
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(100deg, var(--teal-dark), var(--teal));
        border: 0; color: white; box-shadow: 0 5px 12px rgba(15,157,145,.18);
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { transform: translateY(-1px); border-color: var(--teal); }
    div[data-testid="stExpander"] { background: rgba(255,255,255,.9); border: 1px solid var(--line); border-radius: 14px; }
    div[data-testid="stForm"] { background: rgba(255,255,255,.78); border: 1px solid var(--line); padding: 20px; border-radius: 18px; }
    div[data-testid="stFileUploader"] { background: rgba(255,255,255,.75); border: 1px dashed #AFC4D6; border-radius: 14px; padding: 10px; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div {
        border-radius: 10px; border-color: #CFDCE8;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #E1E9F2; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏦 BORROWER RISK")
    st.caption("Explainable financial risk monitoring")
    page = st.radio(
        "NAVIGATION",
        ["Individual Risk Assessment", "Borrower Risk Monitoring"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Ensemble Machine Learning • SHAP")

st.markdown(
    """
    <section class="brand-hero">
      <div class="brand-pill">🛡️ SMART CREDIT RISK INSIGHTS</div>
      <h1 class="brand-title">🛡️ Borrower Risk Predictor</h1>
      <p class="brand-subtitle">Explainable ensemble machine learning for early warning and monitoring of borrower financial risk.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='color:#61738A;font-size:.88rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin:0 0 5px 2px'>{page}</div>",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

MODEL_DIR = Path("borrower_risk_model")

@st.cache_resource
def load_models():
    with gzip.open(MODEL_DIR / "random_forest.pkl.gz", "rb") as f:
        rf_model = joblib.load(f)

    xgb_model = joblib.load(MODEL_DIR / "xgboost.pkl")
    gb_model = joblib.load(MODEL_DIR / "gradient_boosting.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))
    shap_background = joblib.load(MODEL_DIR / "shap_background.pkl")

    return (
        rf_model,
        xgb_model,
        gb_model,
        preprocessor,
        threshold,
        shap_background,
    )


try:
    (
        rf_model,
        xgb_model,
        gb_model,
        preprocessor,
        DECISION_THRESHOLD,
        SHAP_BACKGROUND,
    ) = load_models()
except Exception as e:
    st.error(
        "The saved models could not be loaded. Check that all model "
        "files are present in the borrower_risk_model folder and that "
        "the installed package versions are compatible."
    )
    st.exception(e)
    st.stop()


# ============================================================
# MODEL INPUTS AND FRIENDLY DISPLAY NAMES
# ============================================================

MODEL_FEATURES = [
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]

PURPOSES = {
    "All Other": "all_other",
    "Credit Card": "credit_card",
    "Debt Consolidation": "debt_consolidation",
    "Educational": "educational",
    "Home Improvement": "home_improvement",
    "Major Purchase": "major_purchase",
    "Small Business": "small_business",
}

PURPOSE_LABELS = {
    value: label for label, value in PURPOSES.items()
}

FRIENDLY_NAMES = {
    "purpose": "Loan Purpose",
    "int.rate": "Loan Interest Rate",
    "installment": "Monthly Loan Payment",
    "log.annual.inc": "Log Annual Income",
    "dti": "Debt-to-Income Ratio",
    "fico": "Credit Score",
    "days.with.cr.line": "Length of Credit History",
    "revol.bal": "Outstanding Revolving Credit Balance",
    "revol.util": "Credit Utilization",
    "inq.last.6mths": "Credit Inquiries in Last 6 Months",
    "delinq.2yrs": "Late Payments in Last 2 Years",
    "pub.rec": "Public Financial Records",
}

# Human-readable headers used for monitoring CSV files.
CSV_COLUMNS = {
    "Borrower ID": "borrower_id",
    "Month": "month",
    "Loan Purpose": "purpose",
    "Loan Interest Rate (%)": "int.rate",
    "Monthly Loan Payment": "installment",
    "Annual Income": "annual_income",
    "Debt-to-Income Ratio (%)": "dti",
    "Credit Score": "fico",
    "Length of Credit History (Days)": "days.with.cr.line",
    "Outstanding Revolving Credit Balance": "revol.bal",
    "Credit Utilization (%)": "revol.util",
    "Credit Inquiries in Last 6 Months": "inq.last.6mths",
    "Late Payments in Last 2 Years": "delinq.2yrs",
    "Public Financial Records": "pub.rec",
}

NUMERIC_INPUTS = [
    "int.rate",
    "installment",
    "annual_income",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def ensemble_probabilities(processed_data):
    """Return the ensemble probability for the positive class."""
    p_rf = rf_model.predict_proba(processed_data)[:, 1]
    p_xgb = xgb_model.predict_proba(processed_data)[:, 1]
    p_gb = gb_model.predict_proba(processed_data)[:, 1]

    return (p_rf + p_xgb + p_gb) / 3.0


def prepare_model_data(input_data):
    """
    Transform validated raw inputs.
    The app validates missing fields before this function is called.
    """
    missing = [
        feature for feature in MODEL_FEATURES
        if feature not in input_data.columns
    ]

    if missing:
        raise ValueError(
            "Required model features are missing: "
            + ", ".join(missing)
        )

    model_frame = input_data[MODEL_FEATURES].copy()

    # The original model was trained with credit.policy. This field is
    # intentionally not collected in the UI; use the common eligible value.
    if model_frame["credit.policy"].isnull().any():
        model_frame["credit.policy"] = model_frame["credit.policy"].fillna(1)

    # The saved preprocessing pipeline contains training-fitted imputers
    # for optional numeric fields. Required fields are validated separately.
    return preprocessor.transform(model_frame)


def risk_category(probability):
    if probability >= DECISION_THRESHOLD:
        return "Higher Risk"
    return "Lower Risk"


def validate_borrower_values(values):
    """Validate required borrower inputs and any optional values supplied."""
    errors = []
    required = [
        "int.rate", "installment", "annual_income", "dti",
        "fico", "delinq.2yrs"
    ]
    for key in required:
        if values.get(key) is None or str(values.get(key)).strip() == "":
            errors.append(f"{FRIENDLY_NAMES.get(key, key)} is required.")

    if errors:
        return errors

    try:
        numeric = {
            key: float(value)
            for key, value in values.items()
            if key != "purpose" and value is not None and str(value).strip() != ""
        }
    except (ValueError, TypeError):
        return ["Enter valid numbers in the fields you have completed."]

    if numeric["annual_income"] <= 0:
        errors.append("Annual income must be greater than zero.")
    if not 0 <= numeric["int.rate"] <= 100:
        errors.append("Loan interest rate must be between 0 and 100.")
    if numeric["installment"] < 0:
        errors.append("Monthly loan payment cannot be negative.")
    if numeric["dti"] < 0:
        errors.append("Debt-to-Income Ratio cannot be negative.")
    if not 300 <= numeric["fico"] <= 850:
        errors.append("Credit Score must be between 300 and 850.")
    if numeric["delinq.2yrs"] < 0 or not numeric["delinq.2yrs"].is_integer():
        errors.append("Late Payments in Last 2 Years must be a non-negative whole number.")

    optional_rules = {
        "days.with.cr.line": ("Length of Credit History cannot be negative.", 0, None),
        "revol.bal": ("Revolving credit balance cannot be negative.", 0, None),
        "revol.util": ("Credit Utilization must be between 0 and 100.", 0, 100),
        "inq.last.6mths": ("Recent Credit Inquiries must be a non-negative whole number.", 0, None),
        "pub.rec": ("Public Financial Records must be a non-negative whole number.", 0, None),
    }
    for key, (message, lower, upper) in optional_rules.items():
        if key in numeric:
            value = numeric[key]
            if value < lower or (upper is not None and value > upper):
                errors.append(message)
            if key in ("inq.last.6mths", "pub.rec") and not value.is_integer():
                errors.append(message)

    if values.get("purpose") not in PURPOSE_LABELS:
        errors.append("Select a valid Loan Purpose.")
    return errors


# ============================================================
# SHAP EXPLANATIONS
# ============================================================

def get_friendly_feature_name(feature_name):
    name = str(feature_name)

    if name.startswith("num__"):
        raw_name = name.replace("num__", "", 1)
        return FRIENDLY_NAMES.get(raw_name, raw_name)

    if name.startswith("cat__"):
        raw_name = name.replace("cat__", "", 1)

        if raw_name.startswith("purpose_"):
            purpose_value = raw_name.replace("purpose_", "", 1)
            purpose_label = PURPOSE_LABELS.get(
                purpose_value,
                purpose_value.replace("_", " ").title(),
            )
            return f"Loan Purpose: {purpose_label}"

        return raw_name.replace("_", " ").title()

    return FRIENDLY_NAMES.get(name, name.replace("_", " ").title())


def explain_prediction(processed_row):
    """
    Explain how transformed features influence the ensemble's
    estimated probability for the positive class.
    """
    try:
        background = np.asarray(SHAP_BACKGROUND)

        if background.ndim != 2:
            st.warning(
                "The saved SHAP background data has an unexpected shape. "
                "The risk prediction is available, but the explanation "
                "could not be generated."
            )
            return

        processed_row = np.asarray(processed_row)

        if processed_row.ndim == 1:
            processed_row = processed_row.reshape(1, -1)

        if background.shape[1] != processed_row.shape[1]:
            st.warning(
                "The SHAP background data does not match the model's "
                "processed features. The risk prediction is available, "
                "but the explanation could not be generated."
            )
            return

        # Use a small background sample to reduce computation time.
        if len(background) > 30:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(background), size=30, replace=False
            )
            background = background[indices]

        def prediction_function(data):
            return ensemble_probabilities(np.asarray(data))

        explainer = shap.KernelExplainer(
            prediction_function,
            background,
        )

        shap_values = explainer.shap_values(
            processed_row,
            nsamples=100,
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[0]

        if shap_values.ndim == 2:
            shap_values = shap_values[0]

        feature_names = list(
            preprocessor.get_feature_names_out()
        )

        if len(feature_names) != len(shap_values):
            st.warning(
                "The explanation feature names do not match the "
                "generated SHAP values."
            )
            return

        explanation = pd.DataFrame({
            "Factor": [
                get_friendly_feature_name(name)
                for name in feature_names
            ],
            "SHAP Impact": shap_values,
        })

        # Combine multiple encoded columns under the same
        # user-friendly label when appropriate.
        explanation = (
            explanation.groupby("Factor", as_index=False)
            ["SHAP Impact"].sum()
        )

        explanation["Effect"] = np.where(
            explanation["SHAP Impact"] > 0,
            "Increases estimated risk",
            np.where(
                explanation["SHAP Impact"] < 0,
                "Reduces estimated risk",
                "Little or no measured effect",
            ),
        )

        explanation["Impact Size"] = (
            explanation["SHAP Impact"].abs()
        )

        explanation = explanation.sort_values(
            "Impact Size", ascending=False
        )

        st.subheader("Why is this borrower considered at risk?")
        st.write(
            "These are the factors that influenced the model's assessment. "
            "Each explanation describes how a borrower's information moved "
            "the model's estimate higher or lower."
        )

        simple_explanations = {
            "Credit Score": "The borrower's credit score influenced the model's assessment based on credit patterns learned during training.",
            "Loan Interest Rate": "The interest rate affects the cost of borrowing and contributed to the model's risk estimate.",
            "Monthly Loan Payment": "The monthly payment represents a regular repayment commitment considered by the model.",
            "Log Annual Income": "The borrower's annual income helps the model assess repayment capacity.",
            "Debt-to-Income Ratio": "This compares existing debt with income and helps indicate the borrower's debt burden.",
            "Length of Credit History": "This shows how much past credit information is available about the borrower.",
            "Outstanding Revolving Credit Balance": "This represents the amount currently owed on revolving credit accounts.",
            "Credit Utilization": "This indicates how much of the available revolving credit the borrower is using.",
            "Credit Inquiries in Last 6 Months": "This reflects recent applications for credit recorded in the borrower's history.",
            "Late Payments in Last 2 Years": "This represents recorded late payments and provides information about repayment history.",
            "Public Financial Records": "This represents certain negative public records included in the borrower's credit data.",
        }

        for _, row in explanation.head(5).iterrows():
            impact = float(row["SHAP Impact"])
            factor = str(row["Factor"])
            description = simple_explanations.get(
                factor,
                "This factor contributed to the model's assessment based on patterns learned from the training data."
            )

            if impact > 0:
                st.markdown(f"**{factor}**  ·  🔴 Increased Risk")
                st.write(description + " Its contribution pushed the estimated risk higher.")
            elif impact < 0:
                st.markdown(f"**{factor}**  ·  🟢 Reduced Risk")
                st.write(description + " Its contribution pushed the estimated risk lower.")
            else:
                st.markdown(f"**{factor}**  ·  ⚪ Little or no measured effect")
                st.write(description + " It had little measurable influence on this prediction.")

        with st.expander("View SHAP impact values"):
            st.dataframe(
                explanation[
                    ["Factor", "Effect", "SHAP Impact"]
                ].head(10),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "These explanations describe the model's learned patterns, "
            "not cause and effect. A factor's direction is specific to "
            "this prediction and does not guarantee repayment outcomes."
        )

    except Exception as e:
        st.warning(
            "The prediction was generated, but the SHAP explanation "
            "could not be displayed."
        )
        st.caption(str(e))


# ============================================================
# INPUT CONVERSION
# ============================================================

def form_values_to_model_input(values):
    annual_income = float(values["annual_income"])

    row = {
        "credit.policy": 1,  # Internal default; not collected from the user.
        "purpose": values["purpose"],
        "int.rate": float(values["int.rate"]) / 100.0,
        "installment": float(values["installment"]),
        "log.annual.inc": float(np.log(annual_income)),
        "dti": float(values["dti"]),
        "fico": float(values["fico"]),
        "days.with.cr.line": (float(values["days.with.cr.line"]) if str(values.get("days.with.cr.line", "")).strip() else np.nan),
        "revol.bal": (float(values["revol.bal"]) if str(values.get("revol.bal", "")).strip() else np.nan),
        "revol.util": (float(values["revol.util"]) if str(values.get("revol.util", "")).strip() else np.nan),
        "inq.last.6mths": (int(float(values["inq.last.6mths"])) if str(values.get("inq.last.6mths", "")).strip() else np.nan),
        "delinq.2yrs": (int(float(values["delinq.2yrs"])) if str(values.get("delinq.2yrs", "")).strip() else np.nan),
        "pub.rec": (int(float(values["pub.rec"])) if str(values.get("pub.rec", "")).strip() else np.nan),
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def monitoring_csv_to_model_input(csv_data):
    """
    Convert the user-friendly monitoring CSV columns to the
    exact model features. Annual income is converted to log income.
    """
    df = csv_data.copy()

    # Accept the documented friendly column names.
    for friendly_name, model_name in CSV_COLUMNS.items():
        if friendly_name in df.columns:
            if model_name in df.columns:
                raise ValueError(
                    f"Both '{friendly_name}' and '{model_name}' "
                    "are present. Keep only one column."
                )
            df = df.rename(columns={friendly_name: model_name})

    required_metadata = ["borrower_id", "month"]

    for col in required_metadata:
        if col not in df.columns:
            raise ValueError(
                f"CSV is missing the required column '{col}'."
            )

    if "annual_income" not in df.columns:
        if "log.annual.inc" in df.columns:
            raise ValueError(
                "For monitoring CSVs, use the 'Annual Income' "
                "column with ordinary income values. Do not supply "
                "'log.annual.inc' directly."
            )
        raise ValueError(
            "CSV is missing the required 'Annual Income' column."
        )

    # Only core borrower fields are mandatory in the monitoring upload.
    required_features = [
        "purpose", "int.rate", "installment", "dti", "fico", "delinq.2yrs"
    ]
    missing_features = [col for col in required_features if col not in df.columns]
    for optional_feature in [
        "days.with.cr.line", "revol.bal", "revol.util",
        "inq.last.6mths", "pub.rec"
    ]:
        if optional_feature not in df.columns:
            df[optional_feature] = np.nan
    if "credit.policy" not in df.columns:
        df["credit.policy"] = 1

    if missing_features:
        readable = [
            FRIENDLY_NAMES.get(col, col)
            for col in missing_features
        ]
        raise ValueError(
            "CSV is missing required features: "
            + ", ".join(readable)
        )

    if df[required_metadata].isnull().any().any():
        raise ValueError(
            "Borrower ID and Month cannot be blank."
        )

    # Parse numeric columns. Empty optional cells become NaN and are
    # handled by the model's saved training-time imputers.
    for col in NUMERIC_INPUTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_numeric = [
        "int.rate", "installment", "annual_income", "dti", "fico", "delinq.2yrs"
    ]
    if df[required_numeric].isnull().any().any():
        raise ValueError(
            "A required numeric field is blank or invalid. Check the CSV and try again."
        )

    if (df["annual_income"] <= 0).any():
        raise ValueError("Annual Income must be greater than zero in every row.")

    # The saved model was trained using a logarithmic income feature.
    df["log.annual.inc"] = np.log(df["annual_income"])

    df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()

    valid_purposes = set(PURPOSES.values())
    invalid_purposes = sorted(
        set(df["purpose"]) - valid_purposes
    )

    if invalid_purposes:
        raise ValueError(
            "Unrecognized Loan Purpose value(s): "
            + ", ".join(invalid_purposes)
            + ". Use the purpose values in the downloadable template."
        )

    if not df["month"].astype(str).str.strip().ne("").all():
        raise ValueError("Month cannot be blank.")

    df["month"] = pd.to_datetime(
        df["month"], errors="coerce"
    )

    if df["month"].isnull().any():
        raise ValueError(
            "Month values must be valid dates, for example 2026-01-01."
        )

    if df.duplicated(["borrower_id", "month"]).any():
        raise ValueError(
            "Each borrower can have only one record per month."
        )

    return df


# ============================================================
# CSV TEMPLATE
# ============================================================

def create_csv_template():
    columns = list(CSV_COLUMNS.keys())

    example = {
        "Borrower ID": "B001",
        "Month": "2026-01-01",
            "Loan Purpose": "debt_consolidation",
        "Loan Interest Rate (%)": 12.0,
        "Monthly Loan Payment": 250.0,
        "Annual Income": 60000.0,
        "Debt-to-Income Ratio (%)": 15.0,
        "Credit Score": 700,
        "Length of Credit History (Days)": "",
        "Outstanding Revolving Credit Balance": "",
        "Credit Utilization (%)": "",
        "Credit Inquiries in Last 6 Months": "",
        "Late Payments in Last 2 Years": 0,
        "Public Financial Records": "",
    }

    return pd.DataFrame([example], columns=columns)


# ============================================================
# SIMULATED MONITORING DATA
# ============================================================

def generate_sample_monitoring_data():
    """
    Generate explicitly simulated data for demonstrating the
    monitoring workflow. It does not represent real borrowers.
    """
    rng = np.random.default_rng(42)
    rows = []

    borrower_profiles = [
        {
            "id": "DEMO001",
            "income": 72000,
            "fico": 730,
            "dti": 12,
            "util": 25,
            "balance": 3000,
            "rate": 10.5,
            "installment": 220,
            "inquiries": 0,
            "late": 0,
            "history": 5200,
        },
        {
            "id": "DEMO002",
            "income": 52000,
            "fico": 680,
            "dti": 22,
            "util": 45,
            "balance": 7000,
            "rate": 14.0,
            "installment": 310,
            "inquiries": 1,
            "late": 0,
            "history": 3500,
        },
        {
    "id": "DEMO003",
    "income": 30000,
    "fico": 590,
    "dti": 35,
    "util": 90,
    "balance": 14000,
    "rate": 22.0,
    "installment": 450,
    "inquiries": 5,
    "late": 3,
    "history": 1200,
},
        {
            "id": "DEMO004",
            "income": 85000,
            "fico": 750,
            "dti": 10,
            "util": 20,
            "balance": 2500,
            "rate": 9.5,
            "installment": 200,
            "inquiries": 0,
            "late": 0,
            "history": 6000,
        },
        {
    "id": "DEMO005",
    "income": 36000,
    "fico": 610,
    "dti": 32,
    "util": 85,
    "balance": 12000,
    "rate": 20.0,
    "installment": 420,
    "inquiries": 4,
    "late": 2,
    "history": 1500,
},
    ]

    months = pd.date_range(
        end=pd.Timestamp.today().normalize().replace(day=1),
        periods=6,
        freq="MS",
    )

    for profile in borrower_profiles:
        for i, month in enumerate(months):
            # Small, plausible month-to-month changes.
            # The model determines the resulting risk score.
            income = max(
                12000,
                profile["income"] * (1 + 0.002 * i)
            )
            fico = int(
                np.clip(
                    profile["fico"] + rng.integers(-3, 4) - i,
                    300, 850
                )
            )
            dti = max(
                0,
                profile["dti"] + 0.7 * i
                + rng.normal(0, 0.5)
            )
            utilization = float(
                np.clip(
                    profile["util"] + 1.5 * i
                    + rng.normal(0, 1.5),
                    1, 99
                )
            )
            balance = max(
                0,
                profile["balance"] * (1 + 0.025 * i)
                + rng.normal(0, 150)
            )

            rows.append({
                "borrower_id": profile["id"],
                "month": month,
                "credit.policy": 1,
                "purpose": "debt_consolidation",
                "int.rate": profile["rate"] / 100.0,
                "installment": profile["installment"],
                "log.annual.inc": np.log(income),
                "dti": dti,
                "fico": fico,
                "days.with.cr.line": (
                    profile["history"] + i * 30
                ),
                "revol.bal": balance,
                "revol.util": utilization,
                "inq.last.6mths": min(
                    10, profile["inquiries"] + (1 if i >= 4 else 0)
                ),
                "delinq.2yrs": profile["late"],
                "pub.rec": 0,
            })

    return pd.DataFrame(rows)


def model_data_to_friendly_csv(model_df):
    """Convert internal model data into understandable CSV columns."""
    df = model_df.copy()

    df["Loan Purpose"] = df["purpose"].map(PURPOSE_LABELS)
    df["Loan Interest Rate (%)"] = df["int.rate"] * 100
    df["Annual Income"] = np.exp(df["log.annual.inc"])

    output = pd.DataFrame({
        "Borrower ID": df["borrower_id"],
        "Month": pd.to_datetime(df["month"]).dt.strftime("%Y-%m-%d"),
        "Loan Purpose": df["purpose"].map(PURPOSE_LABELS),
        "Loan Interest Rate (%)": df["Loan Interest Rate (%)"],
        "Monthly Loan Payment": df["installment"],
        "Annual Income": df["Annual Income"],
        "Debt-to-Income Ratio (%)": df["dti"],
        "Credit Score": df["fico"],
        "Length of Credit History (Days)": df["days.with.cr.line"],
        "Outstanding Revolving Credit Balance": df["revol.bal"],
        "Credit Utilization (%)": df["revol.util"],
        "Credit Inquiries in Last 6 Months": df["inq.last.6mths"],
        "Late Payments in Last 2 Years": df["delinq.2yrs"],
        "Public Financial Records": df["pub.rec"],
    })

    return output


# ============================================================
# TABS
# ============================================================

# ============================================================
# PAGE 1: INDIVIDUAL RISK ASSESSMENT
# ============================================================

if page == "Individual Risk Assessment":
    st.subheader("Assess an Individual Borrower")
    st.write(
        "Enter the borrower's main financial details. Additional credit "
        "information is optional and can be expanded when available."
    )

    if st.button("Load Example Borrower", key="load_example"):
        st.session_state["example_loaded"] = True

    example_loaded = st.session_state.get("example_loaded", False)
    if example_loaded:
        st.info("Example borrower details are loaded. You can edit them before assessment.")

    with st.form("borrower_form"):
        st.markdown("### Main Borrower Details")
        c1, c2 = st.columns(2)

        with c1:
            purpose_label = st.selectbox(
                "Loan Purpose", options=list(PURPOSES.keys()),
                index=2 if example_loaded else 0,
            )
            interest_rate_text = st.text_input(
                "Loan Interest Rate (%)", value="14" if example_loaded else "",
                placeholder="Example: 12",
            )
            installment_text = st.text_input(
                "Monthly Loan Payment", value="310" if example_loaded else "",
                placeholder="Example: 250",
            )
            annual_income_text = st.text_input(
                "Annual Income", value="52000" if example_loaded else "",
                placeholder="Example: 60000",
            )

        with c2:
            dti_text = st.text_input(
                "Debt-to-Income Ratio (%)", value="22" if example_loaded else "",
                placeholder="Example: 15",
            )
            fico_text = st.text_input(
                "Credit Score", value="680" if example_loaded else "",
                placeholder="Example: 700",
            )
            late_payments_text = st.text_input(
                "Late Payments in Last 2 Years", value="0" if example_loaded else "",
                placeholder="Example: 0",
            )

        st.markdown("### Additional Credit Information")
        st.caption("These five fields are optional. Leave them blank if the information is unavailable.")
        st.info(
            "Credit Policy Eligibility is not collected in this form. "
            "To remain compatible with the existing trained model, the app "
            "uses its internal default value (Eligible = Yes). This assumption "
            "can influence the prediction."
        )

        with st.expander("Add optional credit details", expanded=False):
            o1, o2 = st.columns(2)
            with o1:
                credit_history_text = st.text_input(
                    "Length of Credit History (Days)",
                    value="3500" if example_loaded else "",
                    placeholder="Optional, e.g. 4000",
                )
                revolving_balance_text = st.text_input(
                    "Outstanding Revolving Credit Balance",
                    value="7000" if example_loaded else "",
                    placeholder="Optional, e.g. 5000",
                )
                utilization_text = st.text_input(
                    "Credit Utilization (%)",
                    value="45" if example_loaded else "",
                    placeholder="Optional, e.g. 30",
                )
            with o2:
                inquiries_text = st.text_input(
                    "Credit Inquiries in Last 6 Months",
                    value="1" if example_loaded else "",
                    placeholder="Optional, e.g. 1",
                )
                public_records_text = st.text_input(
                    "Public Financial Records",
                    value="0" if example_loaded else "",
                    placeholder="Optional, e.g. 0",
                )

        submitted = st.form_submit_button(
            "Assess Borrower Risk", type="primary", use_container_width=True
        )

    if submitted:
        converted_values = {
            "purpose": PURPOSES[purpose_label],
            "int.rate": interest_rate_text,
            "installment": installment_text,
            "annual_income": annual_income_text,
            "dti": dti_text,
            "fico": fico_text,
            "delinq.2yrs": late_payments_text,
            "days.with.cr.line": credit_history_text,
            "revol.bal": revolving_balance_text,
            "revol.util": utilization_text,
            "inq.last.6mths": inquiries_text,
            "pub.rec": public_records_text,
        }
        errors = validate_borrower_values(converted_values)
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                model_input = form_values_to_model_input(converted_values)
                processed = prepare_model_data(model_input)
                probability = float(ensemble_probabilities(processed)[0])
                category = risk_category(probability)

                st.divider()
                st.subheader("Borrower Risk Assessment Result")
                r1, r2 = st.columns(2)
                r1.metric("Estimated Risk Score", f"{probability:.1%}")
                r2.metric("Risk Classification", category)

                if probability >= DECISION_THRESHOLD:
                    st.warning(
                        "This borrower's estimated risk is at or above the "
                        "model's decision threshold. Review the financial "
                        "details before deciding on next steps."
                    )
                else:
                    st.success(
                        "This borrower's estimated risk is below the model's "
                        "decision threshold. This does not guarantee repayment."
                    )

                optional_missing = [
                    name for name, value in {
                        "Length of Credit History": credit_history_text,
                        "Revolving Credit Balance": revolving_balance_text,
                        "Credit Utilization": utilization_text,
                        "Recent Credit Inquiries": inquiries_text,
                        "Public Financial Records": public_records_text,
                    }.items() if not str(value).strip()
                ]
                if optional_missing:
                    st.info(
                        "Some optional details were not provided. The saved "
                        "training-time preprocessing fills these missing "
                        "values using its learned imputation values."
                    )

                st.caption(
                    "This is a model estimate, not a lending decision or a "
                    "validated probability of default within a specific period."
                )
                explain_prediction(processed)

            except Exception as e:
                st.error("The prediction could not be completed. Check the input values and saved model files.")
                st.exception(e)


# ============================================================
# TAB 2: BORROWER RISK MONITORING
# ============================================================

# ============================================================
# PAGE 2: BORROWER RISK MONITORING
# ============================================================

if page == "Borrower Risk Monitoring":

    st.subheader("Borrower Risk Monitoring")
    st.write(
        "Monitor changes in borrower risk, identify borrowers "
        "requiring attention, and understand the factors influencing "
        "their estimated risk."
    )

    st.markdown("### 1. Upload or Generate Monitoring Data")

    template = create_csv_template()

    c1, c2 = st.columns([1, 2])

    with c1:
        st.download_button(
            "Download CSV Template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="borrower_monitoring_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.caption(
            "Upload one record per borrower per month. "
            "The CSV should contain borrower ID, month, annual income "
            "and the required financial details."
        )

    source_choice = st.radio(
        "Choose data source",
        [
            "Generate Sample Monitoring Data",
            "Upload Borrower CSV",
        ],
        horizontal=True,
    )

    monitoring_df = None

    if source_choice == "Generate Sample Monitoring Data":

        if st.button(
            "Generate Sample Data",
            type="primary",
        ):
            try:
                monitoring_df = generate_sample_monitoring_data()
                st.session_state["monitoring_data"] = monitoring_df
                st.session_state["monitoring_source"] = "simulated"
            except Exception as e:
                st.error("Could not generate sample data.")
                st.exception(e)

        if (
            "monitoring_data" in st.session_state
            and st.session_state.get("monitoring_source") == "simulated"
        ):
            monitoring_df = st.session_state["monitoring_data"]

        if monitoring_df is not None:
            st.info(
                "These are simulated borrower records for demonstration. "
                "They do not represent actual bank customers."
            )

    else:

        uploaded_file = st.file_uploader(
            "Upload monthly borrower CSV",
            type=["csv"],
            help="Upload the completed borrower monitoring CSV template.",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                monitoring_df = monitoring_csv_to_model_input(
                    uploaded_df
                )

                st.session_state["monitoring_data"] = monitoring_df
                st.session_state["monitoring_source"] = "uploaded"

                st.success("CSV uploaded and validated successfully.")

            except Exception as e:
                st.error(f"Could not use this CSV: {e}")
                monitoring_df = None

    if monitoring_df is not None:

        try:
            monitoring_df = monitoring_df.copy()

            required_monitoring = (
                ["borrower_id", "month"] + MODEL_FEATURES
            )

            missing = [
                col for col in required_monitoring
                if col not in monitoring_df.columns
            ]

            if missing:
                st.error(
                    "Monitoring data is missing required fields: "
                    + ", ".join(missing)
                )
                st.stop()

            # Optional numeric fields may contain missing values.
            # The saved preprocessing pipeline handles these using
            # training-time imputation.
            required_input_fields = [
                "credit.policy",
                "purpose",
                "int.rate",
                "installment",
                "log.annual.inc",
                "dti",
                "fico",
                "delinq.2yrs",
            ]

            if monitoring_df[required_input_fields].isnull().any().any():
                st.error(
                    "Some required borrower information is missing. "
                    "Please check the uploaded records."
                )
                st.stop()

            monitoring_df = monitoring_df.sort_values(
                ["borrower_id", "month"]
            ).reset_index(drop=True)

            # Prepare model inputs and generate ensemble predictions.
            processed = prepare_model_data(
                monitoring_df[MODEL_FEATURES]
            )

            monitoring_df["risk_probability"] = (
                ensemble_probabilities(processed)
            )

            monitoring_df["risk_classification"] = (
                monitoring_df["risk_probability"].apply(risk_category)
            )

            # Calculate change compared with the previous recorded month.
            monitoring_df["risk_change"] = (
                monitoring_df.groupby("borrower_id")[
                    "risk_probability"
                ].diff()
            )

            # Early-warning conditions:
            # 1. Risk increased by at least 10 percentage points.
            # 2. Current risk score reached the classification threshold.
            monitoring_df["early_warning"] = (
                (monitoring_df["risk_change"] >= 0.10)
                |
                (
                    monitoring_df["risk_probability"]
                    >= DECISION_THRESHOLD
                )
            )

            # Explain why each alert was generated.
            monitoring_df["alert_reason"] = np.select(
                [
                    (
                        (monitoring_df["risk_change"] >= 0.10)
                        &
                        (
                            monitoring_df["risk_probability"]
                            >= DECISION_THRESHOLD
                        )
                    ),
                    monitoring_df["risk_change"] >= 0.10,
                    (
                        monitoring_df["risk_probability"]
                        >= DECISION_THRESHOLD
                    ),
                ],
                [
                    "Risk increased and crossed the threshold",
                    "Risk increased by at least 10 percentage points",
                    "Current risk score is above the threshold",
                ],
                default="No alert",
            )

            # Get the most recent record for each borrower.
            latest_records = (
                monitoring_df.groupby("borrower_id", sort=False)
                .tail(1)
                .copy()
            )

            # Describe each borrower's recent trend.
            def get_risk_trend(change):
                if pd.isna(change):
                    return "Not enough history"
                elif change >= 0.02:
                    return "Increasing"
                elif change <= -0.02:
                    return "Decreasing"
                return "Stable"

            latest_records["risk_trend"] = (
                latest_records["risk_change"].apply(get_risk_trend)
            )

            latest_records["status"] = np.where(
                latest_records["early_warning"],
                "Needs Attention",
                "Lower Risk",
            )

            # ====================================================
            # 2. OVERALL MONITORING SUMMARY
            # ====================================================

            st.divider()
            st.subheader("2. Overall Monitoring Summary")

            total_borrowers = len(latest_records)

            attention_count = int(
                latest_records["early_warning"].sum()
            )

            increasing_count = int(
                (
                    latest_records["risk_trend"] == "Increasing"
                ).sum()
            )

            decreasing_count = int(
                (
                    latest_records["risk_trend"] == "Decreasing"
                ).sum()
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Total Borrowers",
                total_borrowers,
            )

            m2.metric(
                "Need Attention",
                attention_count,
            )

            m3.metric(
                "Risk Increasing",
                increasing_count,
            )

            m4.metric(
                "Risk Decreasing",
                decreasing_count,
            )

            st.caption(
                "The summary uses the latest available record for "
                "each borrower. Risk classifications are based on "
                "the saved model threshold."
            )

            # ====================================================
            # 3. BORROWER RISK OVERVIEW
            # ====================================================

            st.divider()
            st.subheader("3. Borrower Risk Overview")

            st.write(
                "This table shows the latest estimated risk and "
                "recent trend for each borrower."
            )

            overview_df = latest_records[
                [
                    "borrower_id",
                    "risk_probability",
                    "risk_trend",
                    "risk_classification",
                    "status",
                ]
            ].copy()

            overview_df["risk_probability"] = (
                overview_df["risk_probability"].map(
                    lambda value: f"{value:.1%}"
                )
            )

            overview_df = overview_df.rename(
                columns={
                    "borrower_id": "Borrower ID",
                    "risk_probability": "Estimated Risk",
                    "risk_trend": "Risk Trend",
                    "risk_classification": "Risk Classification",
                    "status": "Monitoring Status",
                }
            )

            st.dataframe(
                overview_df,
                use_container_width=True,
                hide_index=True,
            )

            # ====================================================
            # 4. BORROWER RISK TREND GRAPH
            # ====================================================

            st.divider()
            st.subheader("4. Borrower Risk Trend")

            borrower_ids = sorted(
                monitoring_df["borrower_id"].astype(str).unique()
            )

            selected_borrower = st.selectbox(
                "Select a borrower to view their monthly risk trend",
                borrower_ids,
                key="monitoring_borrower_select",
            )

            borrower_history = monitoring_df[
                monitoring_df["borrower_id"].astype(str)
                == selected_borrower
            ].sort_values("month")

            fig, ax = plt.subplots(figsize=(10, 4.5))

            chart_dates = pd.to_datetime(
                borrower_history["month"]
            )

            chart_scores = (
                borrower_history["risk_probability"] * 100
            )

            ax.plot(
                chart_dates,
                chart_scores,
                marker="o",
                linewidth=2.5,
                markersize=7,
                label="Estimated Risk",
            )

            ax.axhline(
                y=DECISION_THRESHOLD * 100,
                linestyle="--",
                linewidth=1.8,
                label=f"Risk Threshold ({DECISION_THRESHOLD:.0%})",
            )

            ax.set_xlabel("Monitoring Month")
            ax.set_ylabel("Estimated Risk Score (%)")
            ax.set_title(
                f"Monthly Risk Trend - {selected_borrower}"
            )

            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "An upward line indicates increasing estimated risk. "
                "A downward line indicates decreasing estimated risk. "
                "The dashed line represents the model's risk threshold."
            )

            latest_borrower = borrower_history.iloc[-1]

            trend_col1, trend_col2, trend_col3 = st.columns(3)

            trend_col1.metric(
                "Latest Estimated Risk",
                f"{latest_borrower['risk_probability']:.1%}",
            )

            trend_col2.metric(
                "Risk Classification",
                latest_borrower["risk_classification"],
            )

            change_value = latest_borrower["risk_change"]

            trend_col3.metric(
                "Change Since Previous Month",
                (
                    "First record"
                    if pd.isna(change_value)
                    else f"{change_value:+.1%}"
                ),
            )

            # ====================================================
            # 5. WHY IS THIS BORROWER AT RISK?
            # ====================================================

            st.divider()
            st.subheader("5. Understand the Risk Factors")

            st.write(
                "Generate an explanation for the selected borrower's "
                "latest risk assessment using SHAP."
            )

            selected_index = borrower_history.index[-1]

            if st.button(
                "Explain Selected Borrower's Risk",
                type="primary",
                key="monitoring_shap_button",
            ):
                st.markdown(
                    f"### Risk Explanation: {selected_borrower}"
                )

                st.metric(
                    "Latest Estimated Risk",
                    f"{latest_borrower['risk_probability']:.1%}",
                )

                st.write(
                    "The following factors show how the borrower's "
                    "financial information influenced the model's "
                    "latest risk estimate."
                )

                explain_prediction(
                    processed[selected_index]
                )

            # ====================================================
            # 6. EARLY WARNING ALERTS
            # ====================================================

            st.divider()
            st.subheader("6. Early Warning Alerts")

            st.write(
                "These alerts identify borrowers who need further "
                "review based on their latest available records."
            )

            latest_alerts = latest_records[
                latest_records["early_warning"]
            ].sort_values(
                "risk_probability",
                ascending=False,
            )

            if latest_alerts.empty:

                st.success(
                    "No borrowers currently meet the early-warning "
                    "conditions in their latest records."
                )

            else:

                for _, alert in latest_alerts.iterrows():

                    with st.container(border=True):

                        alert_col1, alert_col2 = st.columns(
                            [3, 1]
                        )

                        with alert_col1:
                            st.markdown(
                                f"**Borrower ID: {alert['borrower_id']}**"
                            )

                            st.write(
                                f"Month: {pd.to_datetime(alert['month']).strftime('%b %Y')}"
                            )

                            st.write(
                                f"Reason: {alert['alert_reason']}"
                            )

                        with alert_col2:
                            st.metric(
                                "Estimated Risk",
                                f"{alert['risk_probability']:.1%}",
                            )

                        if pd.notna(alert["risk_change"]):
                            st.caption(
                                "Change since previous month: "
                                f"{alert['risk_change']:+.1%}"
                            )

                        st.caption(
                            "Review the borrower's financial information. "
                            "An alert is not an automatic lending decision."
                        )

            # ====================================================
            # 7. DETAILED MONTHLY RECORDS
            # ====================================================

            st.divider()

            with st.expander(
                "View Detailed Monthly Monitoring Records",
                expanded=False,
            ):

                display_df = monitoring_df.copy()

                display_df["month"] = pd.to_datetime(
                    display_df["month"]
                ).dt.strftime("%Y-%m-%d")

                display_df["risk_probability"] = (
                    display_df["risk_probability"].map(
                        lambda value: f"{value:.1%}"
                    )
                )

                display_df["risk_change"] = (
                    display_df["risk_change"].map(
                        lambda value: (
                            "—"
                            if pd.isna(value)
                            else f"{value:+.1%}"
                        )
                    )
                )

                display_df["early_warning"] = (
                    display_df["early_warning"].map(
                        {
                            True: "Alert",
                            False: "No Alert",
                        }
                    )
                )

                display_df = display_df.rename(
                    columns={
                        "borrower_id": "Borrower ID",
                        "month": "Month",
                        "risk_probability": "Estimated Risk",
                        "risk_classification": "Risk Classification",
                        "risk_change": "Change Since Previous Month",
                        "early_warning": "Early Warning",
                    }
                )

                st.dataframe(
                    display_df[
                        [
                            "Borrower ID",
                            "Month",
                            "Estimated Risk",
                            "Risk Classification",
                            "Change Since Previous Month",
                            "Early Warning",
                            "alert_reason",
                        ]
                    ].rename(
                        columns={
                            "alert_reason": "Alert Reason"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # ====================================================
            # 8. DOWNLOAD ANALYSIS
            # ====================================================

            st.divider()
            st.subheader("7. Download Monitoring Analysis")

            st.write(
                "Download the borrower risk scores, monthly changes, "
                "classifications and early-warning information."
            )

            download_df = monitoring_df.copy()

            download_df["month"] = pd.to_datetime(
                download_df["month"]
            ).dt.strftime("%Y-%m-%d")

            download_df["risk_probability"] = (
                download_df["risk_probability"] * 100
            )

            download_df["risk_change"] = (
                download_df["risk_change"] * 100
            )

            download_df = download_df.rename(
                columns={
                    "borrower_id": "Borrower ID",
                    "month": "Month",
                    "risk_probability": "Estimated Risk Score (%)",
                    "risk_classification": "Risk Classification",
                    "risk_change": "Change Since Previous Month (%)",
                    "early_warning": "Early Warning",
                    "alert_reason": "Alert Reason",
                }
            )

            st.download_button(
                "Download Risk Analysis CSV",
                data=download_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="borrower_risk_analysis.csv",
                mime="text/csv",
                type="primary",
            )

            st.caption(
                "Risk scores are model estimates based on historical "
                "training patterns. Alerts indicate records requiring "
                "review, not confirmed future defaults."
            )

        except Exception as e:

            st.error(
                "Monitoring could not be completed. "
                "Check the uploaded data and model compatibility."
            )

            st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic prototype for demonstrating ensemble-based borrower "
    "risk estimation, model explanations, and monitoring. It is not "
    "a production lending system. Predictions and alerts require "
    "further validation before real-world use."
)
