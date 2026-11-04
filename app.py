
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from evaluate import compute_metrics

TARGET = "grade_class"
DROP_COLS = ["final_grade_pct", "at_risk_flag", "student_id", "survey_date"]   # ignored if present
MODEL_FILES = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
               "KNN": "knn", "Naive Bayes (Gaussian)": "naive_bayes",
               "Support Vector Machine (SVM)": "svm"}

# Page configuration
st.set_page_config(
    page_title="ML assignment 2 -2026AJ05043 ",
    page_icon="🤖",
    layout="centered"
)


# =====================================
# APPLICATION HEADER
# =====================================

st.title("🤖 Building and Comparing Performance of Machine Learning Models")
st.caption("A framework for managing ML models.")

st.divider()

# =====================================
# SIDEBAR NAVIGATION
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Model Configuration",
        "Model Execution",
        "Results"
    ]
)

# =====================================
# HOME
# =====================================

if page == "Home":

    st.header("Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Available Models", 0)
    col2.metric("Executed Models", 0)
    col3.metric("Application Status", "Ready")

    st.divider()

    st.subheader("Application Overview")

    st.info(
        "Welcome to the ML Application Framework. "
        "Model integration and execution functionality "
        "will be added in future releases."
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption("ML Application Framework | Built with Python and Streamlit")