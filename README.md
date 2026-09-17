# Loan-Approval-Predictor
# 🏦 Loan Approval Predictor

A Machine Learning and Streamlit web application that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.

## 🚀 Features

* Data preprocessing
* Exploratory Data Analysis
* Logistic Regression
* Model evaluation
* Loan approval prediction
* Approval probability
* Interactive Streamlit UI

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## 📂 Project Structure

```text
Loan-Approval-Predictor/
│
├── loan_approval.ipynb
├── loan_approval_dataset.csv
├── loan_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

## 🧠 Model

The project uses **Logistic Regression** with:

* `StandardScaler` for numerical features
* `OneHotEncoder` for categorical features
* `Pipeline` for preprocessing and prediction

## 📊 Prediction

The application predicts:

```text
✅ Loan Approved
or
❌ Loan Rejected
```

It also displays the **approval and rejection probability**.

## 👨‍💻 Author

**Anuj Mahore**
B.Tech CSE | IIIT Bhopal

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used for actual financial decisions.
