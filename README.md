# Student Grade Class Prediction – ML Assignment 2 - 2026AJ05043

## a. Problem statement
Predict a secondary-school student's **letter grade class (A, B, C, D or F)** from socioeconomic and behavioural survey information (family income, parents' education and occupation, residence, internet access, study hours, attendance, tutoring, parental involvement, previous-year grade, etc.).
This is a **multi-class classification** problem. Five classification models – Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine – are trained on the same dataset and compared using Accuracy, AUC, Precision, Recall, F1 score and Matthews Correlation Coefficient (MCC). The trained models are served through an interactive Streamlit web app.

## b. Dataset description
| Item | Details |
|---|---|
| Name | `student_grade_ses_dataset.csv` – synthetic cross-sectional survey of secondary-school students (grades 6–12; urban, semi-urban and rural households). Generated from published distribution shapes and effect directions; contains no real student data. |
| Size | 12,025 rows × 29 columns; 12,000 rows after removing 25 duplicate `student_id` records |
| Target (column AB) | `grade_class` – 5 classes: **C** 6,147 (51.2%), **D** 2,740 (22.8%), **B** 2,686 (22.4%), **F** 218 (1.8%), **A** 209 (1.7%) → imbalanced |
| Columns dropped | `final_grade_pct` (column AA – the numeric score the letter grade is derived from, i.e. target leakage), `at_risk_flag` (column AC), and the identifiers `student_id`, `survey_date` |
| Features used | 24 features (14 numeric, 10 categorical) |
| Missing values | 1.5%–6% in nine columns (income, parents' education/occupation, internet access, distance, attendance, previous-year grade) |

**Feature groups**
- *Demographic / school:* age, gender, grade_level, school_type, academic_year
- *Household / socioeconomic:* residence_type, family_size, num_siblings, single_parent_household, family_income_bracket, annual_family_income_inr, father/mother education level, father/mother occupation, has_internet_access, has_computer_or_laptop, scholarship_status
- *Behavioural / academic:* distance_to_school_km, extra_tutoring, parental_involvement_score, study_hours_per_week, attendance_rate_pct, previous_year_grade_pct

**Preprocessing:** median imputation + standard scaling for numeric features; most-frequent imputation + one-hot encoding for categorical features (fitted on training data only and stored inside each saved model pipeline). 80/20 stratified train/test split, `random_state=42` → 9,600 training and 2,400 test rows (`test_data.csv`).

## c. GitHub repository link
- **GitHub repository:** `https://github.com/mrobo00/py_ml_assign_2`
- **Live Streamlit app:** `https://pymlassign2-euetcxq8kehc5rnfbrncrj.streamlit.app/`

## d. Models used
All five models were trained on the same training split and evaluated on the same 2,400-row hold-out test set. Because the target has five classes, **AUC** is one-vs-rest, macro-averaged; **Precision, Recall and F1** are weighted averages over classes; **MCC** is the multi-class Matthews correlation coefficient.

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5812 | 0.7233 | 0.5701 | 0.5812 | 0.5511 | 0.2837 |
| Decision Tree | 0.5692 | 0.7279 | 0.5503 | 0.5692 | 0.5386 | 0.2619 |
| KNN | 0.5433 | 0.6583 | 0.5314 | 0.5433 | 0.5008 | 0.2016 |
| Naive Bayes (Gaussian) | 0.3667 | 0.6604 | 0.4604 | 0.3667 | 0.3477 | 0.1676 |
| Support Vector Machine (SVM) | 0.5792 | 0.7222 | 0.5811 | 0.5792 | 0.5367 | 0.2711 |

Settings: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=6, min_samples_leaf=10`), KNN (`k=15`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `probability=True`).

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best accuracy (0.581), F1 (0.551) and MCC (0.284), with train accuracy (0.592) ≈ test accuracy (0.581), so it generalises well. The signal is largely linear – `previous_year_grade_pct` is by far the strongest coefficient – so a simple, interpretable model is hard to beat. It predicts the majority class C well (recall 0.81) but almost never predicts A or F. |
| Decision Tree | Highest AUC (0.728) and second-best MCC/F1; limiting depth to 6 keeps it from overfitting (train 0.593 vs test 0.569). It relies mainly on `previous_year_grade_pct` (≈70% of importance) and `annual_family_income_inr` (≈17%). Its hard predictions are slightly worse than Logistic Regression's. |
| KNN | Weaker than the linear models (accuracy 0.543, MCC 0.202) and has the lowest AUC of all five models (0.658). After one-hot encoding there are many binary features that dilute the distance information, and k=15 mostly votes for the majority class C. |
| Naive Bayes (Gaussian) | Clearly the worst (accuracy 0.367, MCC 0.168). Its Gaussian and feature-independence assumptions do not suit one-hot and strongly correlated socioeconomic features, so it over-predicts class D. It is, however, the only model that recovers a meaningful share of the rare class A (recall 0.60), at the price of many false alarms. |
| Support Vector Machine (SVM) | Very close to Logistic Regression (accuracy 0.579, AUC 0.722) and the best weighted precision (0.581), but it fits the training data more tightly (train 0.668 vs test 0.579, mild overfitting), is much slower to train, and adds no accuracy over a linear boundary. |

**Overall winner for this dataset: Logistic Regression.** It has the best accuracy, F1 and MCC, a near-top AUC (0.723, within 0.005 of the Decision Tree's best), no overfitting, and is the simplest and fastest model. SVM and Decision Tree are close runners-up.

*Caveat:* the dataset is highly imbalanced (A and F are ~2% each) and the features only moderately explain the grade, so every model reaches only ~55–58% accuracy and MCC stays below 0.3. Per-class recall for A and F is near zero for all models except Naive Bayes (A). Class weighting / resampling and hyper-parameter tuning would be the next steps.

## Project structure
```
app.py                  Streamlit app (loads saved models; no retraining)
generate_models.py      CREATES the .joblib files: trains the 5 models, evaluates, saves, verifies
data_prep.py            loading, cleaning, split, preprocessing pipeline
evaluate.py             the six evaluation metrics
models_def/             one .py per classifier definition (+ __init__.py registry)
model/                  saved models (.joblib): logistic_regression, decision_tree, knn,
                        naive_bayes, svm, label_encoder  (+ model_info.json with library versions)
test_data.csv           hold-out test set (2,400 rows, features + target) - upload in the app
metrics.csv             comparison table above
requirements.txt        dependencies for Streamlit Cloud
README.md               this document
student_grade_ses_dataset.csv        full dataset (training only; the app never reads it)
```
## How the .joblib model files are generated
Run once (about 30-60 seconds):

```
python generate_models.py
```

1.	loads the dataset, drops columns AA / AC / identifiers, removes duplicates and makes the 80/20 stratified split;
2.	writes the 20% hold-out part to test_data.csv;
3.	trains each classifier inside a pipeline;
4.	evaluates every model with the six metrics and writes metrics.csv;
5.	saves each pipeline with joblib.dump to model/<name>.joblib, plus model/label_encoder.joblib and model/model_info.json;
6.	reloads every file and checks that its predictions are identical to the in-memory model.
Because the whole pipeline is saved, the app can take a raw CSV row and predict without retraining.


## Streamlit app features
1. **Dataset upload (CSV)** – upload the test data (must contain all input features and the `grade_class` target column).
2. **Model selection dropdown** – Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM.
3. **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1, MCC for the selected model, plus a table with the results of **all five models** on the uploaded test data.
4. **Confusion matrix and classification report** for the selected model (plus ROC curves and downloadable predictions).
5. **Predict one student** – optional form for single predictions.

The app loads the saved models from `model/` and only predicts – it never retrains on the uploaded data.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
The saved models were created with scikit-learn 1.9.1 (pinned in `requirements.txt`); rerun `train_evaluate.py` if you use a different version.
