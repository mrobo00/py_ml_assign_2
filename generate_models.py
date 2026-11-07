"""
generate_models.py  -  creates ALL the saved-model (.joblib) files used by the Streamlit app.

Run it once (about 30-60 seconds, the SVM is the slow one):

        python generate_models.py

What it does, step by step
--------------------------
  1. loads the dataset, drops column AA / AC / ids, removes duplicates and splits 80 % / 20 %   (data_prep.py)
  2. writes the 20 % hold-out part (features + true label) to  test_data.csv  -> upload it in the app
  3. builds each of the 5 classifiers                                                        (models_def/)
     and trains it INSIDE a pipeline  [imputer -> scaler / one-hot -> classifier]
  4. evaluates every model on the hold-out set with the 6 metrics                              (evaluate.py)
  5. SAVES every trained pipeline with joblib into the  model/  folder:
          model/logistic_regression.joblib      model/decision_tree.joblib      model/knn.joblib
          model/naive_bayes.joblib              model/svm.joblib
          model/label_encoder.joblib            (maps 0..4  <->  'A'..'F')
  6. writes  metrics.csv  (comparison table)  and  model/model_info.json  (versions + file list)
  7. VERIFIES the files: reloads each .joblib from disk and checks its predictions are identical
     to the model that is still in memory.

Why save a whole *pipeline* and not just the classifier?
    The classifier only understands numbers that were scaled / one-hot encoded in a specific way.
    The pipeline remembers those preprocessing steps, so the app can pass in a raw CSV row and
    get a prediction - WITHOUT retraining anything.
"""
import json
import os

import joblib                      # the standard way to save scikit-learn objects to disk
import numpy as np
import pandas as pd
import sklearn

from data_prep import get_split, make_pipeline, save_test_data
from evaluate import compute_metrics
from models_def import MODELS

MODEL_DIR = "model"                # folder that will contain the .joblib files

# File name used for each model on disk (app.py uses the same mapping to load them).
SAVE_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes (Gaussian)": "naive_bayes",
    "Support Vector Machine (SVM)": "svm",
}


def fit_all(X_tr, y_tr):
    """Train every model in MODELS on the training data. Returns {model name: fitted pipeline}."""
    fitted = {}
    for name, build_model in MODELS.items():
        print(f"  training {name} ...")
        # make_pipeline() = preprocessing + a fresh classifier; .fit() learns BOTH
        # (imputer medians, scaler means/stds, one-hot categories and the classifier itself) from X_tr only.
        fitted[name] = make_pipeline(build_model(), X_tr).fit(X_tr, y_tr)
    return fitted


def evaluate_all(fitted, X_te, y_te):
    """Compute the 6 metrics for every fitted pipeline on the hold-out set -> DataFrame (one row per model)."""
    rows = []
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_te)            # hard class predictions (0..4)
        y_proba = pipe.predict_proba(X_te)     # class probabilities, needed for AUC
        rows.append({"Model": name, **compute_metrics(y_te, y_pred, y_proba)})
    return pd.DataFrame(rows).set_index("Model").round(4)


def save_models(fitted, label_encoder, model_dir=MODEL_DIR):
    """Write every fitted pipeline + the label encoder to  <model_dir>/*.joblib  and return the file paths."""
    os.makedirs(model_dir, exist_ok=True)      # create model/ if it does not exist yet
    paths = {}
    for name, pipe in fitted.items():
        path = os.path.join(model_dir, f"{SAVE_NAMES[name]}.joblib")
        joblib.dump(pipe, path)                # <- this single line creates the .joblib file
        paths[name] = path
    # The label encoder is saved too: the app needs it to turn predictions 0..4 back into 'A'..'F'.
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    return paths


def verify_saved_models(fitted, paths, X_te):
    """Reload every .joblib file from disk and confirm it predicts exactly like the in-memory model."""
    for name, path in paths.items():
        reloaded = joblib.load(path)
        same = np.array_equal(reloaded.predict(X_te), fitted[name].predict(X_te))
        size_kb = os.path.getsize(path) / 1024
        print(f"  {os.path.basename(path):28s} {size_kb:9.1f} KB   reload gives identical predictions: {same}")
        assert same, f"{path} does not reproduce the trained model's predictions!"


def main():
    print("1/6  Writing the hold-out test set -> test_data.csv")
    save_test_data("test_data.csv")

    print("2/6  Loading data and splitting into train / test")
    X_tr, X_te, y_tr, y_te, le = get_split()
    print(f"     train rows: {len(X_tr)}   test rows: {len(X_te)}   classes: {list(le.classes_)}")

    print("3/6  Training the 5 models")
    fitted = fit_all(X_tr, y_tr)

    print("4/6  Evaluating on the hold-out set")
    results = evaluate_all(fitted, X_te, y_te)
    results.to_csv("metrics.csv")              # the comparison table that goes into the README
    print(results.to_string())

    print(f"5/6  Saving models to ./{MODEL_DIR}/")
    paths = save_models(fitted, le)

    print("6/6  Verifying the saved files")
    verify_saved_models(fitted, paths, X_te)

    # A small text record of how the files were made. The scikit-learn version matters because
    # .joblib files can only be loaded reliably with the same version (see requirements.txt).
    info = {"scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "classes": list(le.classes_),
            "model_files": {n: os.path.basename(p) for n, p in paths.items()},
            "test_rows": int(len(X_te))}
    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("\nDone. Next step:  streamlit run app.py")


if __name__ == "__main__":      # only run when executed as a script, not when imported
    main()
