import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant details to predict the loan approval status.")

st.divider()

# Input form
with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=0,
            value=500000,
            step=10000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=300000,
            step=10000
        )

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            max_value=30,
            value=10,
            step=1
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=750,
            step=1
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=0,
            value=500000,
            step=10000
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            value=200000,
            step=10000
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=0,
            value=100000,
            step=10000
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=0,
            value=200000,
            step=10000
        )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )


# Prediction
if submitted:

    # Create DataFrame with exactly the same feature names
    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education],
        "self_employed": [self_employed],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction probability
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    probability_dict = dict(zip(classes, probabilities))

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == "Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    approval_probability = probability_dict.get("Approved", 0) * 100
    rejection_probability = probability_dict.get("Rejected", 0) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Rejection Probability",
            f"{rejection_probability:.2f}%"
        )

    st.progress(int(approval_probability))

    with st.expander("View Applicant Details"):
        st.dataframe(input_data, use_container_width=True)
