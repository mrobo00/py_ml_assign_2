"""
models_def  -  the definition (algorithm + hyper-parameters) of each classifier.

Each model lives in its own small file with:
    NAME        - the display name used in tables, the app drop-down and file names
    get_model() - returns a fresh, UNtrained scikit-learn estimator

MODELS below links the five files together: {display name: function that builds the model}.
generate_models.py loops over this dictionary to train and save every model.
"""
from . import logistic_regression, decision_tree, knn, naive_bayes, svm

MODELS = {m.NAME: m.get_model for m in
          (logistic_regression, decision_tree, knn, naive_bayes, svm)}
