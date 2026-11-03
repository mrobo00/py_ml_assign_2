"""Data loading, cleaning, splitting and preprocessing (shared by every other file)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

DATA_FILE = "student_grade_ses_dataset.csv"
TARGET = "grade_class"                       # column AB
DROP_COLS = ["final_grade_pct",              # column AA (leaks the target)
             "at_risk_flag",                 # column AC
             "student_id", "survey_date"]    # identifiers
RANDOM_STATE = 42


def load_data(path=DATA_FILE):
    df = pd.read_csv(path).drop_duplicates("student_id")
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def get_split(path=DATA_FILE, test_size=0.2):
    """Returns X_train, X_test, y_train, y_test (encoded), LabelEncoder."""
    df = load_data(path)
    le = LabelEncoder()
    y = le.fit_transform(df.pop(TARGET))
    X_tr, X_te, y_tr, y_te = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    return X_tr, X_te, y_tr, y_te, le


def build_preprocessor(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])


def make_pipeline(estimator, X):
    return Pipeline([("prep", build_preprocessor(X)), ("clf", estimator)])


def save_test_data(path="test_data.csv"):
    X_tr, X_te, y_tr, y_te, le = get_split()
    t = X_te.copy(); t[TARGET] = le.inverse_transform(y_te)
    t.to_csv(path, index=False)
    return t
