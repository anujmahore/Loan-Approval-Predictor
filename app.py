import streamlit as st
import pandas as pd
import pickle

try:
    with open("loan_model.pkl", "rb") as file:
        package = pickle.load(file)

    if isinstance(package, dict) and "model" in package:
        model = package["model"]
        training_ranges = package.get("training_ranges", {})
    else:
        model = package
        training_ranges = {}

except FileNotFoundError:
    st.error("loan_model.pkl not found. Run the Jupyter notebook first.")
    st.stop()

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant details to predict the loan approval status.")
st.divider()

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents", 0, 10, 2, 1
        )

        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed", ["Yes", "No"]
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=0,
            max_value=100_000_000,
            value=5_000_000,
            step=10_000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            max_value=100_000_000,
            value=3_000_000,
            step=10_000
        )

        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=2,
            max_value=360,
            value=120,
            step=1
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score", 300, 900, 750, 1
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=0,
            max_value=100_000_000,
            value=5_000_000,
            step=10_000
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            max_value=100_000_000,
            value=2_000_000,
            step=10_000
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=0,
            max_value=100_000_000,
            value=1_000_000,
            step=10_000
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=0,
            max_value=100_000_000,
            value=2_000_000,
            step=10_000
        )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education.strip()],
        "self_employed": [self_employed.strip()],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })

    # Warn when an input is outside the training distribution.
    out_of_range = []
    for col, limits in training_ranges.items():
        value = float(input_data[col].iloc[0])
        if value < limits["min"] or value > limits["max"]:
            out_of_range.append(
                f"{col}: {value:,.0f} "
                f"(training range {limits['min']:,.0f} - {limits['max']:,.0f})"
            )

    if out_of_range:
        st.warning(
            "⚠️ Some values are outside the range seen during training. "
            "The prediction may be unreliable."
        )
        with st.expander("View out-of-range values"):
            for item in out_of_range:
                st.write("•", item)

    prediction = str(model.predict(input_data)[0]).strip()
    probabilities = model.predict_proba(input_data)[0]

    probability_dict = {
        str(cls).strip(): float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    approval = probability_dict.get("Approved", 0.0) * 100
    rejection = probability_dict.get("Rejected", 0.0) * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == "Approved":
        st.success("✅ Loan Approved")
    elif prediction == "Rejected":
        st.error("❌ Loan Rejected")
    else:
        st.info(f"Prediction: {prediction}")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Approval Probability", f"{approval:.2f}%")

    with c2:
        st.metric("Rejection Probability", f"{rejection:.2f}%")

    st.progress(max(0, min(100, int(round(approval)))))

    with st.expander("View Applicant Details"):
        st.dataframe(input_data, use_container_width=True)
