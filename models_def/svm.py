"""Model 5 - Support Vector Machine (RBF kernel)."""
from sklearn.svm import SVC

NAME = "Support Vector Machine (SVM)"


def get_model():
    # kernel="rbf"       : non-linear decision boundary
    # probability=True   : also fit an internal calibration so predict_proba() works.
    #                      The AUC metric and the app's probability chart need probabilities.
    #                      (This makes training slower - it runs an internal 5-fold cross-validation.)
    # random_state       : makes that internal calibration reproducible
    return SVC(kernel="rbf", probability=True, random_state=42)
