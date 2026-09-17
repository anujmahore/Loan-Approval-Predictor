import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Load dataset and trained model
# --------------------------------------------------
try:
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    with open("loan_model.pkl", "rb") as file:
        package = pickle.load(file)

    # Supports the latest notebook format
    if isinstance(package, dict) and "model" in package:
        model = package["model"]
    else:
        model = package

except FileNotFoundError as e:
    st.error(
        "Required file not found. Keep `loan_approval_dataset.csv`, "
        "`loan_model.pkl`, and `app.py` in the same folder."
    )
    st.stop()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Predictor")
st.write(
    "Enter applicant details. The input ranges below are automatically "
    "taken from the training dataset."
)

st.divider()


# --------------------------------------------------
# Get exact values/ranges from dataset
# --------------------------------------------------
numeric_columns = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

# Helper for safe defaults
def median_value(column):
    return int(round(df[column].median()))

def min_value(column):
    return int(df[column].min())

def max_value(column):
    return int(df[column].max())


# --------------------------------------------------
# Input form
# --------------------------------------------------
with st.form("loan_form"):

    st.subheader("👤 Applicant Information")

    col1, col2 = st.columns(2)

    with col1:

        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=min_value("no_of_dependents"),
            max_value=max_value("no_of_dependents"),
            value=median_value("no_of_dependents"),
            step=1,
            help=f"Allowed range: {min_value('no_of_dependents')} - {max_value('no_of_dependents')}"
        )

        education_values = sorted(df["education"].dropna().unique().tolist())

        education = st.selectbox(
            "Education",
            education_values,
            help="Select one of the education categories present in the dataset."
        )

        self_employed_values = sorted(
            df["self_employed"].dropna().unique().tolist()
        )

        self_employed = st.selectbox(
            "Self Employed",
            self_employed_values,
            help="Select one of the categories present in the dataset."
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=min_value("income_annum"),
            max_value=max_value("income_annum"),
            value=median_value("income_annum"),
            step=10000,
            help=f"Allowed range: ₹{min_value('income_annum'):,} - ₹{max_value('income_annum'):,}"
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=min_value("loan_amount"),
            max_value=max_value("loan_amount"),
            value=median_value("loan_amount"),
            step=10000,
            help=f"Allowed range: ₹{min_value('loan_amount'):,} - ₹{max_value('loan_amount'):,}"
        )

        # IMPORTANT:
        # The Kaggle dataset represents loan_term using values such as 2–20.
        # Do NOT convert this field to 120 for a 10-year term.
        loan_term = st.number_input(
            "Loan Term",
            min_value=min_value("loan_term"),
            max_value=max_value("loan_term"),
            value=median_value("loan_term"),
            step=1,
            help=f"Use the same representation as the Kaggle dataset. Allowed range: {min_value('loan_term')} - {max_value('loan_term')}"
        )

    with col2:

        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=min_value("cibil_score"),
            max_value=max_value("cibil_score"),
            value=median_value("cibil_score"),
            step=1,
            help=f"Allowed range: {min_value('cibil_score')} - {max_value('cibil_score')}"
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=min_value("residential_assets_value"),
            max_value=max_value("residential_assets_value"),
            value=median_value("residential_assets_value"),
            step=10000,
            help=f"Allowed range: ₹{min_value('residential_assets_value'):,} - ₹{max_value('residential_assets_value'):,}"
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=min_value("commercial_assets_value"),
            max_value=max_value("commercial_assets_value"),
            value=median_value("commercial_assets_value"),
            step=10000,
            help=f"Allowed range: ₹{min_value('commercial_assets_value'):,} - ₹{max_value('commercial_assets_value'):,}"
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=min_value("luxury_assets_value"),
            max_value=max_value("luxury_assets_value"),
            value=median_value("luxury_assets_value"),
            step=10000,
            help=f"Allowed range: ₹{min_value('luxury_assets_value'):,} - ₹{max_value('luxury_assets_value'):,}"
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=min_value("bank_asset_value"),
            max_value=max_value("bank_asset_value"),
            value=median_value("bank_asset_value"),
            step=10000,
            help=f"Allowed range: ₹{min_value('bank_asset_value'):,} - ₹{max_value('bank_asset_value'):,}"
        )

    st.caption(
        "💡 Tip: Use values within the displayed ranges. "
        "These ranges are taken directly from the Kaggle training dataset."
    )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

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

    prediction = str(model.predict(input_data)[0]).strip()

    probabilities = model.predict_proba(input_data)[0]

    probability_dict = {
        str(cls).strip(): float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    approval_probability = probability_dict.get("Approved", 0.0) * 100
    rejection_probability = probability_dict.get("Rejected", 0.0) * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == "Approved":
        st.success("✅ Loan Approved")
    elif prediction == "Rejected":
        st.error("❌ Loan Rejected")
    else:
        st.info(f"Prediction: {prediction}")

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

    st.progress(
        max(0, min(100, int(round(approval_probability))))
    )

    with st.expander("👤 View Applicant Details"):
        st.dataframe(input_data, use_container_width=True)
