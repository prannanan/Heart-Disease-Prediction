# Heart Disease Prediction

**Super AI Engineer Season 6 – Hackathon 5**  
**Metric: F2-Score** (recall-weighted; critical in healthcare settings)

---

## Problem Statement

Given a large-scale health survey dataset (~100k+ rows), predict whether a patient has a **history of heart disease or attack** (`Yes` / `No`). The evaluation metric is **F2-score**, which penalizes false negatives (missing sick patients) more heavily than false positives — appropriate for medical risk prediction.

---

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | Labeled training data |
| `test.csv` | Unlabeled test data for submission |
| `sample_submission.csv` | Submission format reference |
| `submission.csv` | Final generated predictions |

**Target column:** `History of HeartDisease or Attack` (`Yes` / `No`)

### Key Features

| Type | Features |
|------|----------|
| Binary (Yes/No) | High Blood Pressure, Told High Cholesterol, Cholesterol Checked, Smoked 100+ Cigarettes, Diagnosed Stroke, Diagnosed Diabetes, Leisure Physical Activity, Heavy Alcohol Consumption, Health Care Coverage, Doctor Visit Cost Barrier, Difficulty Walking, Vegetable or Fruit Intake (1+ per Day) |
| Binary | Sex (Male / Female) |
| Continuous | Body Mass Index (BMI) |
| Ordinal | Age, General Health (Very Poor → Excellent), Education Level, Income Level |

---

## Pipeline

```
1. Data loading & inspection
2. EDA (class balance, binary features, continuous features)
3. Preprocessing & encoding
4. Feature engineering
5. Correlation analysis
6. Model training with 5-fold stratified cross-validation
7. SMOTE experiment
8. Threshold tuning for F2-score
9. OOF evaluation & confusion matrix
10. Feature importance analysis
11. Final model retraining & submission generation
```

---

## Preprocessing

- **Drop:** `ID` column (non-predictive identifier)
- **Binary encoding:** Yes/No columns → `1`/`0`; `Sex` → `Male=1, Female=0`
- **Ordinal encoding:**
  - `General Health`: Very Poor=1 … Excellent=6
  - `Education Level`: Never attended=1 … College graduate=6
  - `Income Level`: <$10k=1 … ≥$75k=8
- **Missing value imputation:** Filled with training set median (no data leakage)

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `risk_count` | Sum of 5 high-risk binary conditions (HBP, High Cholesterol, Stroke, Diabetes, Walking Difficulty) |
| `bmi_cat` | BMI binned into 4 categories: underweight / normal / overweight / obese |
| `lifestyle_score` | Sum of unhealthy lifestyle indicators (smoking, heavy drinking, inactivity, poor diet) |

---

## Models Evaluated

All models use `class_weight='balanced'` to boost recall on the minority positive class.

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble; robust to outliers |
| **HistGradientBoostingClassifier** | Best performer; handles missing values natively |

Cross-validation: **5-Fold Stratified K-Fold**, scored with `fbeta_score(beta=2)`.

---

## Class Imbalance Handling

Two strategies were compared:

- **`class_weight='balanced'`** — re-weights the loss function to penalize minority-class errors more
- **SMOTE** (`sampling_strategy=0.5`) — generates synthetic minority-class samples in feature space

The better-performing strategy was selected for final training.

---

## Threshold Tuning

Instead of the default 0.5 probability threshold:

1. Collected **out-of-fold probabilities** (unbiased estimates)
2. Swept thresholds from `0.05` to `0.75` in steps of `0.01`
3. Selected the threshold that **maximized OOF F2-score**

Lowering the threshold increases recall (catches more sick patients) at the cost of more false positives — an acceptable trade-off given the F2 metric.

---

## Results

- **Best model:** `HistGradientBoostingClassifier` (max_iter=300, lr=0.05, max_depth=6)
- **Optimized classification threshold** selected via OOF F2 sweep
- Evaluation via classification report + confusion matrix on OOF predictions

---

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

2. Update `DATA_PATH` in the notebook to point to your local dataset directory.

3. Run all cells in `Heart Disease Prediction.ipynb` sequentially.

4. The final submission is saved to `super-ai-engineer-ss-6-heart-disease-prediction/submission.csv`.

---

## Project Structure

```
Heart Disease Prediction/
├── Heart Disease Prediction.ipynb   # Main notebook
├── instructions.md                  # Competition brief (Thai)
├── README.md                        # This file
└── super-ai-engineer-ss-6-heart-disease-prediction/
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    └── submission.csv
```
