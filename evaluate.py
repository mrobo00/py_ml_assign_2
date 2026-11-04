"""Evaluation metrics (shared by the notebook, training script and Streamlit app)."""
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_proba):
    """AUC: one-vs-rest macro (NaN if the labels don't contain all classes).
    Precision / Recall / F1: weighted average (multi-class)."""
    try:
        auc_score = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc_score = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_score,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
