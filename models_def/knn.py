"""Model 3 - K-Nearest Neighbors classifier."""
from sklearn.neighbors import KNeighborsClassifier

NAME = "KNN"


def get_model():
    # n_neighbors=15: each prediction is a vote among the 15 most similar training students.
    # A fairly large k smooths out noise; k=1 would simply copy the closest student.
    # (Distances are meaningful because the preprocessing standardises all numeric features.)
    return KNeighborsClassifier(n_neighbors=15)
