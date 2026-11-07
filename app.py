
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


tab1, tab2 = st.tabs(["🧪 Evaluate on test data", "🧍 Predict one student"])

# ------------------------------------------------ tab 1: upload + evaluate
with tab1:
    up = st.file_uploader("Upload test data (CSV). It must contain all input features and the target column "
                          f"`{TARGET}` – e.g. the provided `test_data.csv`.", type="csv")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False, disabled=up is not None)
    df = pd.read_csv(up) if up is not None else (pd.read_csv("test_data.csv") if use_sample else None)

    if df is None:
        st.info("Upload a CSV (or tick the checkbox) to see results.")
    else:
        missing = [c for c in feature_cols + [TARGET] if c not in df]
        bad = [] if TARGET in missing else sorted(set(df[TARGET].dropna().unique()) - set(classes))
        if missing:
            st.error(f"The uploaded file is missing required columns: {missing}")
        elif bad:
            st.error(f"`{TARGET}` has unexpected values {bad}; expected only {classes}.")
        else:
            X = df[feature_cols]
            y = le.transform(df[TARGET])
            st.write(f"**{len(df):,} rows loaded.**")

            # results for every model on the uploaded test data
            preds, probas, rows = {}, {}, {}
            for n, p in models.items():
                preds[n], probas[n] = p.predict(X), p.predict_proba(X)
                rows[n] = compute_metrics(y, preds[n], probas[n])
            allres = pd.DataFrame(rows).T.round(4); allres.index.name = "ML Model"

            st.subheader("Evaluation metrics – all models on the uploaded test data")
            st.dataframe(allres.style.highlight_max(axis=0, color="#b7e4c7").format("{:.4f}", na_rep="n/a"),
                         use_container_width=True)
            st.caption("AUC = one-vs-rest macro (needs all 5 classes present) · Precision/Recall/F1 = weighted average")

            st.subheader(f"Selected model: {choice}")
            for col, (k, v) in zip(st.columns(6), rows[choice].items()):
                col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

            a, b = st.columns(2)
            with a:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y, preds[choice], labels=range(len(classes))), annot=True, fmt="d",
                            cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix – {choice}")
                st.pyplot(fig)
            with b:
                st.markdown("**Classification report**")
                st.code(classification_report(y, preds[choice], labels=range(len(classes)),
                                              target_names=classes, zero_division=0))

            with st.expander("ROC curves (one-vs-rest) for the selected model"):
                if len(set(y)) == len(classes):
                    Y = label_binarize(y, classes=range(len(classes)))
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for i, c in enumerate(classes):
                        fpr, tpr, _ = roc_curve(Y[:, i], probas[choice][:, i])
                        ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
                    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
                    st.pyplot(fig)
                else:
                    st.write("ROC needs all 5 classes in the uploaded labels.")

            out = df.copy(); out["predicted_grade_class"] = le.inverse_transform(preds[choice])
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ------------------------------------------------- tab 2: single prediction
with tab2:
    st.subheader(f"Predict one student's grade class with: {choice}")
    ref = pd.read_csv("test_data.csv")[feature_cols]          # only used for input ranges / defaults
    num_cols = ref.select_dtypes("number").columns
    values, cols = {}, st.columns(3)
    for i, c in enumerate(feature_cols):
        with cols[i % 3]:
            if c in num_cols:
                s = ref[c].dropna()
                if (s % 1 == 0).all():
                    values[c] = st.number_input(c, int(s.min()), int(s.max()), int(s.median()), step=1)
                else:
                    values[c] = st.number_input(c, float(s.min()), float(s.max()), float(round(s.median(), 2)))
            else:
                opts = sorted(ref[c].dropna().unique().tolist())
                values[c] = st.selectbox(c, opts, index=opts.index(ref[c].mode()[0]))
    if st.button("Predict", type="primary"):
        p = models[choice].predict_proba(pd.DataFrame([values])[feature_cols])[0]
        st.success(f"Predicted grade class: **{classes[int(p.argmax())]}** (confidence {p.max():.1%})")
        st.bar_chart(pd.Series(p, index=classes, name="probability"))