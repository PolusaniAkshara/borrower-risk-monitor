import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="Borrower Financial Risk Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Borrower Financial Risk Monitor")
st.write(
    "Estimate the risk that a borrower may not fully repay a loan "
    "using an ensemble of machine learning models."
)

@st.cache_resource
def load_models():
    rf = joblib.load("borrower_risk_model/random_forest.pkl")
    xgb = joblib.load("borrower_risk_model/xgboost.pkl")
    gb = joblib.load("borrower_risk_model/gradient_boosting.pkl")
    preprocessor = joblib.load("borrower_risk_model/preprocessor.pkl")
    threshold = joblib.load("borrower_risk_model/threshold.pkl")
    background = joblib.load("borrower_risk_model/shap_background.pkl")
    return rf, xgb, gb, preprocessor, threshold, background

rf, xgb, gb, preprocessor, threshold, X_background = load_models()

st.subheader("Enter Borrower Details")

with st.form("borrower_form"):
    col1, col2 = st.columns(2)

    with col1:
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "all_other", "credit_card", "debt_consolidation",
                "educational", "home_improvement",
                "major_purchase", "small_business"
            ]
        )
        interest_rate = st.number_input(
            "Interest Rate (e.g. 0.12 = 12%)",
            min_value=0.0, max_value=1.0, value=0.12, step=0.01
        )
        installment = st.number_input(
            "Monthly Installment ($)", min_value=0.0, value=250.0
        )
        annual_income = st.number_input(
            "Annual Income ($)", min_value=0.0, value=60000.0
        )
        dti = st.number_input(
            "Debt-to-Income Ratio (%)", min_value=0.0, value=15.0
        )
        fico = st.number_input(
            "FICO Credit Score", min_value=300, max_value=850, value=700
        )
        credit_history = st.number_input(
            "Days with Credit Line", min_value=0.0, value=4000.0
        )

    with col2:
        revolving_balance = st.number_input(
            "Revolving Credit Balance ($)", min_value=0.0, value=5000.0
        )
        revolving_utilization = st.number_input(
            "Revolving Credit Utilization (%)",
            min_value=0.0, max_value=100.0, value=30.0
        )
        recent_inquiries = st.number_input(
            "Credit Inquiries in Last 6 Months",
            min_value=0, value=1
        )
        delinquencies = st.number_input(
            "Delinquencies in Last 2 Years", min_value=0, value=0
        )
        public_records = st.number_input(
            "Public Records", min_value=0, value=0
        )
        credit_policy = st.selectbox(
            "Meets Credit Policy?", ["Yes", "No"]
        )

    submitted = st.form_submit_button("Estimate Risk")

if submitted:
    if annual_income <= 0:
        st.error("Annual income must be greater than zero.")
    else:
        borrower = pd.DataFrame([{
            "credit.policy": 1 if credit_policy == "Yes" else 0,
            "purpose": purpose,
            "int.rate": interest_rate,
            "installment": installment,
            "log.annual.inc": np.log(annual_income),
            "dti": dti,
            "fico": fico,
            "days.with.cr.line": credit_history,
            "revol.bal": revolving_balance,
            "revol.util": revolving_utilization,
            "inq.last.6mths": recent_inquiries,
            "delinq.2yrs": delinquencies,
            "pub.rec": public_records
        }])

        processed = preprocessor.transform(borrower)

        probabilities = [
            rf.predict_proba(processed)[0, 1],
            xgb.predict_proba(processed)[0, 1],
            gb.predict_proba(processed)[0, 1]
        ]
        risk_probability = float(np.mean(probabilities))
        prediction = risk_probability >= threshold

        st.subheader("Risk Assessment")
        st.metric("Estimated Risk Probability", f"{risk_probability:.1%}")

        if prediction:
            st.warning(
                "Higher predicted risk: the model estimates an increased "
                "chance that the loan may not be fully repaid."
            )
        else:
            st.success(
                "Lower predicted risk relative to the selected threshold."
            )

        st.caption(
            f"Decision threshold: {threshold:.2f}. "
            "This is a prototype estimate, not a lending decision."
        )

        # Explain this prediction using SHAP
        st.subheader("Why was this risk predicted?")

        try:
            import shap

            background = X_background if "X_background" in globals() else None

            if background is None:
                st.info(
                    "SHAP explanation requires a saved background dataset. "
                    "We will prepare that in the next step."
                )
            else:
                explainer = shap.KernelExplainer(
                    lambda data: (
                        rf.predict_proba(data)[:, 1]
                        + xgb.predict_proba(data)[:, 1]
                        + gb.predict_proba(data)[:, 1]
                    ) / 3,
                    background
                )
                values = explainer.shap_values(
                    processed, nsamples=100
                )
                shap_array = np.array(values).reshape(-1)

                friendly_names = {
    "num__installment": "Monthly Loan Installment",
    "num__log.annual.inc": "Annual Income",
    "num__inq.last.6mths": "Recent Credit Inquiries",
    "num__credit.policy": "Credit Policy Eligibility",
    "num__int.rate": "Loan Interest Rate",
    "num__revol.util": "Credit Utilization",
    "num__days.with.cr.line": "Length of Credit History",
    "num__fico": "Credit Score",
    "num__revol.bal": "Revolving Credit Balance",
    "num__dti": "Debt-to-Income Ratio",
    "num__delinq.2yrs": "Past Delinquencies",
    "num__pub.rec": "Public Records",
    "cat__purpose_all_other": "Purpose: Other",
    "cat__purpose_credit_card": "Purpose: Credit Card",
    "cat__purpose_debt_consolidation": "Purpose: Debt Consolidation",
    "cat__purpose_educational": "Purpose: Education",
    "cat__purpose_home_improvement": "Purpose: Home Improvement",
    "cat__purpose_major_purchase": "Purpose: Major Purchase",
    "cat__purpose_small_business": "Purpose: Small Business"
}

feature_names = [
    friendly_names.get(name, name)
    for name in preprocessor.get_feature_names_out()
]
                importance = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_array
                })
                importance["Impact"] = importance["SHAP Value"].abs()
                importance = importance.nlargest(5, "Impact")

                st.write("Top 5 factors influencing this prediction:")
                st.dataframe(
                    importance[["Feature", "SHAP Value"]],
                    use_container_width=True
                )

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")
