from sklearn.tree import DecisionTreeClassifier
NAME = "Decision Tree"
def get_model():
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
