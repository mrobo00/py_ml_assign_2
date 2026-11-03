from sklearn.linear_model import LogisticRegression
NAME = "Logistic Regression"
def get_model():
    return LogisticRegression(max_iter=2000, random_state=42)
