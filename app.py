"""
app.py  -  Streamlit web app for the Student Grade Class Prediction project.

IMPORTANT: this app does NOT train anything. It loads the models that generate_models.py
saved in the model/ folder (.joblib files) and only calls predict() / predict_proba() on the data
that the user uploads.

Run locally:   streamlit run app.py
Layout:        sidebar  -> model drop-down
               tab 1    -> upload test CSV, see metrics, confusion matrix, classification report, ROC
               tab 2    -> enter the details of ONE student and get a prediction
"""
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from evaluate import compute_metrics       # same metric code that was used at training time

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TARGET = "grade_class"                                              # name of the label column in the CSV
DROP_COLS = ["final_grade_pct", "at_risk_flag", "student_id", "survey_date"]   # ignored if present in an upload
# display name  ->  file name inside model/   (must match SAVE_NAMES in generate_models.py)
MODEL_FILES = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
               "KNN": "knn", "Naive Bayes (Gaussian)": "naive_bayes",
               "Support Vector Machine (SVM)": "svm"}

st.set_page_config(page_title="Student Grade Classifier", page_icon="🎓", layout="wide")


# ----------------------------------------------------------------------------
# Helper: classification report as a complete table
# ----------------------------------------------------------------------------
def report_tables(y_true, y_pred, class_names):
    """
    Build the classification report as TWO complete tables, because they hold different information:

      per_class : one row per grade (A..F)  ->  precision, recall, f1-score, support of THAT grade
      overall   : summaries over ALL grades ->  accuracy, macro average, weighted average

    sklearn's text report mixes both in one list and leaves precision/recall empty in the 'accuracy' row.
    Here the overall accuracy is written into precision, recall and f1-score of its row (for single-label
    multi-class data, micro-averaged precision = recall = f1 = accuracy) and support is the total sample count.
    """
    rep = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0, output_dict=True)
    cols = ["precision", "recall", "f1-score", "support"]
    per_class = pd.DataFrame({c: rep[c] for c in class_names}).T[cols]
    per_class.index.name = "Grade"
    acc = rep["accuracy"]                                        # a single number in sklearn's dict
    total = rep["weighted avg"]["support"]
    overall = pd.DataFrame({
        "Accuracy": {"precision": acc, "recall": acc, "f1-score": acc, "support": total},
        "Macro average": rep["macro avg"],
        "Weighted average": rep["weighted avg"]}).T[cols]
    overall.index.name = "Summary"
    # round to 4 decimals with the same rule as the comparison table, so all numbers on the page agree
    return per_class.round(4), overall.round(4)


# ----------------------------------------------------------------------------
# Load the saved models (once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models…")
def load_models():
    """Read the 5 pipelines and the label encoder from model/. Cached, so reruns of the page are instant."""
    models = {name: joblib.load(f"model/{f}.joblib") for name, f in MODEL_FILES.items()}
    return models, joblib.load("model/label_encoder.joblib")


models, le = load_models()
classes = list(le.classes_)                       # ['A', 'B', 'C', 'D', 'F']

# Find out which raw input columns the models expect. Every saved pipeline starts with a
# ColumnTransformer ("prep") that lists the numeric and categorical columns it was trained on.
prep = next(iter(models.values()))["prep"]
needed = [c for _, _, cols in prep.transformers_ if isinstance(cols, list) for c in cols]
# keep them in the same order as in test_data.csv (just reading the header row)
feature_cols = [c for c in pd.read_csv("test_data.csv", nrows=0).columns if c in needed]

# ----------------------------------------------------------------------------
# Sidebar: model selection drop-down
# ----------------------------------------------------------------------------
st.sidebar.header("Settings")
choice = st.sidebar.selectbox("Select model", list(MODEL_FILES))
st.sidebar.markdown(
    "**Target:** `grade_class` (A–F)  \n"
    "Models are loaded from `model/` – nothing is retrained on the uploaded data.")

st.title("🎓 Student Grade Class Prediction")
st.caption("Upload the test CSV, pick a model, and see the evaluation metrics, confusion matrix and classification report.")

tab1, tab2 = st.tabs(["🧪 Evaluate on test data", "🧍 Predict one student"])

# ============================================================================
# TAB 1 - upload a CSV and evaluate the selected model (and compare all models)
# ============================================================================
with tab1:
    # (a) dataset upload option - only the small test file is uploaded, not the full dataset
    up = st.file_uploader("Upload test data (CSV). It must contain all input features and the target column "
                          f"`{TARGET}` – e.g. the provided `test_data.csv`.", type="csv")
    # convenience: let the user try the app with the bundled file (disabled once a file is uploaded)
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False, disabled=up is not None)
    df = pd.read_csv(up) if up is not None else (pd.read_csv("test_data.csv") if use_sample else None)

    if df is None:
        st.info("Upload a CSV (or tick the checkbox) to see results.")
    else:
        # --- validate the uploaded file -------------------------------------------------
        missing = [c for c in feature_cols + [TARGET] if c not in df]       # required columns not in the file
        bad = [] if TARGET in missing else sorted(set(df[TARGET].dropna().unique()) - set(classes))  # labels outside A-F
        if missing:
            st.error(f"The uploaded file is missing required columns: {missing}")
        elif bad:
            st.error(f"`{TARGET}` has unexpected values {bad}; expected only {classes}.")
        else:
            X = df[feature_cols]                    # model inputs (extra columns such as the ids are ignored)
            y = le.transform(df[TARGET])            # true labels as integers 0..4
            st.write(f"**{len(df):,} rows loaded.**")

            # --- run ALL five saved models on the uploaded data (prediction only, no training) ---
            preds, probas, rows = {}, {}, {}
            for n, p in models.items():
                preds[n], probas[n] = p.predict(X), p.predict_proba(X)
                rows[n] = compute_metrics(y, preds[n], probas[n])
            allres = pd.DataFrame(rows).T.round(4)
            allres.index.name = "ML Model"

            # (c) evaluation metrics of every model side by side (best value per column is highlighted)
            st.subheader("Evaluation metrics – all models on the uploaded test data")
            st.dataframe(allres.style.highlight_max(axis=0, color="#b7e4c7").format("{:.4f}", na_rep="n/a"))
            st.caption("AUC = one-vs-rest macro (needs all 5 classes present) · Precision/Recall/F1 = weighted average")

            # metrics of the model chosen in the sidebar, shown as 6 big numbers
            st.subheader(f"Selected model: {choice}")
            # tiles use the SAME rounded values as the comparison table above (e.g. 0.58125 -> 0.5812 in both)
            for col, (k, v) in zip(st.columns(6), allres.loc[choice].items()):
                col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

            # (d) confusion matrix (left) and ROC curves (right) of the selected model, side by side
            a, b = st.columns(2)
            with a:
                fig, ax = plt.subplots(figsize=(5, 4))
                # labels=range(5) makes the matrix always 5x5 even if a class is missing from the upload
                sns.heatmap(confusion_matrix(y, preds[choice], labels=range(len(classes))), annot=True, fmt="d",
                            cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix – {choice}")
                st.pyplot(fig)
            with b:
                if len(set(y)) == len(classes):                       # ROC needs every class to be present
                    Y = label_binarize(y, classes=range(len(classes)))   # one 0/1 column per class
                    fig, ax = plt.subplots(figsize=(5, 4))
                    for i, c in enumerate(classes):
                        fpr, tpr, _ = roc_curve(Y[:, i], probas[choice][:, i])
                        ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
                    ax.plot([0, 1], [0, 1], "k--", lw=1)             # diagonal = random guessing
                    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
                    ax.set_title("ROC curves (one-vs-rest)"); ax.legend(fontsize=8)
                    st.pyplot(fig)
                else:
                    st.info("ROC curves need all 5 classes in the uploaded labels.")

            # classification report, shown as two full-width tables (static st.table: nothing is cut off)
            per_class, overall = report_tables(y, preds[choice], classes)
            fmt = {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}

            st.subheader("Classification report")
            st.markdown("**1. Per-grade metrics** – one row per grade (A–F), each calculated only for that grade")
            st.table(per_class.style.format(fmt))
            st.caption("**precision** = of the students predicted as this grade, the share that really have it · "
                       "**recall** = of the students who really have this grade, the share the model found · "
                       "**f1-score** = balance of precision and recall · "
                       "**support** = number of students with this grade in the uploaded test data (the true count)")

            st.markdown("**2. Overall summary** – one row for all grades together (not per grade)")
            st.table(overall.style.format(fmt))
            st.caption("**Accuracy** = correct predictions ÷ all students (one number, so it is repeated in each column) · "
                       "**Macro average** = plain mean of the 5 grades, so rare grades A and F count as much as C · "
                       "**Weighted average** = mean weighted by support, so big grades count more "
                       "(these are the Precision / Recall / F1 shown in the tiles above) · "
                       "**support** here = total number of students")

            # offer the predictions of the selected model as a downloadable CSV
            out = df.copy()
            out["predicted_grade_class"] = le.inverse_transform(preds[choice])   # 0..4 -> 'A'..'F'
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ============================================================================
# TAB 2 - predict the grade class of a single student from a form
# ============================================================================
with tab2:
    st.subheader(f"Predict one student's grade class with: {choice}")
    # test_data.csv is used ONLY to decide the allowed range / default of each input widget
    ref = pd.read_csv("test_data.csv")[feature_cols]
    num_cols = ref.select_dtypes("number").columns
    values, cols = {}, st.columns(3)                 # lay the form out in 3 columns
    for i, c in enumerate(feature_cols):
        with cols[i % 3]:
            if c in num_cols:
                s = ref[c].dropna()
                if (s % 1 == 0).all():               # whole-number feature -> integer input, default = median
                    values[c] = st.number_input(c, int(s.min()), int(s.max()), int(s.median()), step=1)
                else:                                # decimal feature -> float input, default = median
                    values[c] = st.number_input(c, float(s.min()), float(s.max()), float(round(s.median(), 2)))
            else:                                    # categorical feature -> drop-down, default = most common value
                opts = sorted(ref[c].dropna().unique().tolist())
                values[c] = st.selectbox(c, opts, index=opts.index(ref[c].mode()[0]))
    if st.button("Predict", type="primary"):
        # one-row DataFrame with the same column order as in training
        p = models[choice].predict_proba(pd.DataFrame([values])[feature_cols])[0]
        st.success(f"Predicted grade class: **{classes[int(p.argmax())]}** (confidence {p.max():.1%})")
        st.bar_chart(pd.Series(p, index=classes, name="probability"))     # probability of each class
