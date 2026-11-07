"""Model 1 - Logistic Regression (multinomial / softmax for the 5 grade classes)."""
from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"


def get_model():
    # max_iter=2000 : the default (100) is too few iterations for the optimiser to converge
    #                 on ~64 scaled + one-hot features, so we allow more.
    # random_state  : fixes any randomness so the saved model is reproducible.
    return LogisticRegression(max_iter=2000, random_state=42)
