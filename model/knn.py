from sklearn.neighbors import KNeighborsClassifier
NAME = "KNN"
def get_model():
    return KNeighborsClassifier(n_neighbors=15)
