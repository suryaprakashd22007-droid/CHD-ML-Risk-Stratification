# Machine Learning-Driven Risk Stratification and Early Detection System for Coronary Heart Disease

An explainable, academic machine learning application for clinical risk stratification and early detection of Coronary Heart Disease (CHD) using the gold-standard Cleveland Clinic dataset.

---

## 1. Project Overview
Coronary Heart Disease (CHD) is among the leading global causes of mortality. This project delivers an end-to-end clinical decision-support system featuring automated data preprocessing, hyperparameter-tuned machine learning models, transparent screening-focused model selection, and an interactive Streamlit dashboard.

---

## 2. Problem Statement
Cardiologists frequently evaluate high-dimensional non-invasive biomarkers (blood pressure, exercise ECG, cholesterol, thallium scintigraphy) to stratify cardiovascular risk. Conventional diagnostic workflows can face diagnostic delay or missed early signals. A predictive screening framework helps identify subtle ischemic patterns while transparently highlighting influential clinical factors.

---

## 3. Objectives
* Build clean, leak-free preprocessing pipelines (handling missing values in `ca` and `thal` via median imputation).
* Implement and benchmark three classifiers: Logistic Regression, Support Vector Machines (SVM), and Random Forest.
* Prioritize **Recall (Sensitivity)** and **ROC-AUC** to minimize dangerous false negatives in a screening setting.
* Provide interactive explanation of model behavior via feature importance rankings.
* Persist patient records and predictions into an auditable SQLite database via SQLAlchemy.

---

## 4. Dataset
* **Source**: Cleveland Clinic Foundation Heart Disease Database (`data/heart-disease-cleveland.csv`).
* **Records**: 303 patient cases.
* **Target Conversion**: Multi-level angiographic disease status `diagnosis` (0 to 4) converted to binary classification `target`:
  * `0`: No Heart Disease (164 cases)
  * `1`: Heart Disease Present (139 cases)

---

## 5. Features (13 Clinical Predictors)
1. `age`: Age in years
2. `sex`: Sex (1 = Male, 0 = Female)
3. `cp`: Chest pain type (1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal Pain, 4 = Asymptomatic)
4. `trestbps`: Resting blood pressure (mm Hg)
5. `chol`: Serum cholesterol (mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
7. `restecg`: Resting ECG results (0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy)
8. `thalach`: Maximum heart rate achieved during exercise
9. `exang`: Exercise-induced angina (1 = Yes, 0 = No)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: Slope of the peak exercise ST segment (1 = Upsloping, 2 = Flat, 3 = Downsloping)
12. `ca`: Number of major vessels (0–3) colored by fluoroscopy
13. `thal`: Thallium defect (3.0 = Normal, 6.0 = Fixed Defect, 7.0 = Reversible Defect)

---

## 6. Data Preprocessing & Pipelines
* Missing values indicated by `?` in `ca` and `thal` are converted to `np.nan` and imputed using `SimpleImputer(strategy='median')`.
* To prevent data leakage, imputation and standard scaling (`StandardScaler`) are encapsulated directly inside `sklearn.pipeline.Pipeline` objects.
* Train/Test split: 80% training (242 samples), 20% testing (61 samples) using stratified sampling (`stratify=target`).

---

## 7. Machine Learning Models & Hyperparameter Tuning
Optimized using **Stratified 5-Fold Cross Validation** via `GridSearchCV` scoring on recall:
1. **Logistic Regression** (L2 penalty, tuned `C` parameter, Standard Scaled)
2. **Support Vector Machine (SVC)** (RBF kernel, tuned `C` and `gamma`, Standard Scaled)
3. **Random Forest Classifier** (Tuned `n_estimators`, `max_depth`, `min_samples_split`, median imputed)

### Actual Benchmark Results on Held-out Test Set (61 Patients):
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.9180** | **0.8710** | **0.9643** | **0.9153** | **0.9535** |
| Logistic Regression | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9513 |
| Support Vector Machine | 0.8197 | 0.7297 | 0.9643 | 0.8308 | 0.9058 |

**Champion Selection Rationale**: Random Forest achieved the top Recall (0.9643) and highest ROC-AUC (0.9535) and Accuracy (0.9180), minimizing false negatives.

---

## 8. Application Screens
* **Dashboard**: Key performance indicators, system status, and comparative metrics snapshot.
* **Patient Risk Prediction**: Clinical parameter entry form, probability estimation, risk outcome, and dynamic feature importance chart.
* **Dataset Analysis**: Descriptive statistics, distribution histograms, box plots, categorical breakdown, and Pearson correlation heatmap.
* **Model Performance**: Full evaluation tables, ROC curves, confusion matrices, and best hyperparameters.
* **Feature Analysis**: Random Forest Gini impurity importance, Logistic Regression weights, and analysis of notable predictors (`thalach`, `cp`, `thal`, `ca`, `oldpeak`).
* **Prediction History**: Auditable table with search by Patient ID, filters, and CSV export.
* **About Project**: Objectives, architectural overview, constraints, and clinical disclaimers.

---

## 9. Database Architecture (SQLite + SQLAlchemy)
Located at `database/heart_disease.db`:
* **`patients`**: Stores `patient_id`, `age`, `sex`, and timestamp.
* **`clinical_biometrics`**: Stores the 13 clinical biomarkers linked to `patient_id`.
* **`risk_assessments`**: Stores `predicted_risk_score`, `predicted_class`, `model_name`, and timestamp.

---

## 10. Installation & Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Machine Learning Models
```bash
python train_models.py
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 11. Limitations & Future Enhancements
* **Sample Size**: Limited to 303 cases from Cleveland clinic. Future work includes testing on Hungarian, Swiss, and Long Beach cohorts.
* **Demographics**: Historical trial data is male-skewed.
* **Explainability Enhancements**: Integration of SHAP (SHapley Additive exPlanations) values for individual local explanations.

---

## 12. Ethical & Clinical Disclaimer
> **IMPORTANT**: This application is an educational machine-learning decision-support prototype. Its output is **not a medical diagnosis** and must not replace evaluation by a licensed healthcare professional.
