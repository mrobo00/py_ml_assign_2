
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

st.set_page_config(page_title="Student Grade Classifier", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading saved models…")
def load_models():
    models = {n: joblib.load(f"model/{f}.joblib") for n, f in MODEL_FILES.items()}
    return models, joblib.load("model/label_encoder.joblib")


models, le = load_models()
classes = list(le.classes_)
# the saved preprocessing step knows which raw input columns the models need
prep = next(iter(models.values()))["prep"]
feature_cols = [c for _, _, cols in prep.transformers_ if isinstance(cols, list) for c in cols]
feature_cols = [c for c in pd.read_csv("test_data.csv", nrows=0).columns if c in feature_cols]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
choice = st.sidebar.selectbox("Select model", list(MODEL_FILES))
st.sidebar.markdown(
    "**Target:** `grade_class` (A–F)  \n"
    "Models are loaded from `model/` – nothing is retrained on the uploaded data.")

st.title("🎓 Student Grade Class Prediction")
st.caption("Upload the test CSV, pick a model, and see the evaluation metrics, confusion matrix and classification report.")
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