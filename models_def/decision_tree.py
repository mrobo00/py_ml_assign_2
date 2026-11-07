"""Model 2 - Decision Tree classifier (CART, Gini impurity)."""
from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"


def get_model():
    # max_depth=6        : limit the tree depth - an unrestricted tree memorises the training data (overfits)
    # min_samples_leaf=10: every leaf must contain at least 10 students, which smooths the predictions
    # random_state       : makes tie-breaking between equally good splits reproducible
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
