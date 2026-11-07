"""Model 4 - Gaussian Naive Bayes."""
from sklearn.naive_bayes import GaussianNB

NAME = "Naive Bayes (Gaussian)"


def get_model():
    # Defaults are used: the model just learns, for every class, the mean and variance of every feature
    # (assuming features are independent and normally distributed given the class).
    # var_smoothing=1e-9 (default) adds a tiny value to variances to avoid division by zero.
    return GaussianNB()
