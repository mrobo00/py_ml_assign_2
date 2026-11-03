from sklearn.svm import SVC
NAME = "SVM (RBF)"
def get_model():
    return SVC(kernel="rbf", probability=True, random_state=42)
