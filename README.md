# Thyroid Cancer Classification: A Comparative ML Study

## Overview

This project builds and compares multiple machine learning models to predict whether a thyroid tumor is malignant or benign based on patient data and risk factors. The dataset contains over 212,000 patient records with features including demographics, medical history, hormone levels, and nodule characteristics.

**The core challenge:** Minimizing false negatives (cancerous tumors misclassified as benign) — the most harmful type of error in medical diagnosis — while maintaining reasonable overall accuracy.

**Key finding:** All models plateau at 82–83% accuracy and ~45% recall for the malignant class, suggesting inherent overlap in the feature space that may make perfect prediction impossible with the available data.

---

## Data Source

Kaggle thyroid cancer dataset containing biopsy records. Features include:

| Category | Examples |
|----------|----------|
| Demographics | Age, Gender, Country, Ethnicity |
| Medical history | Family history, radiation exposure, iodine deficiency, smoking, obesity, diabetes |
| Clinical measures | TSH, T3, T4 levels, nodule size |
| Risk assessment | Thyroid cancer risk score |
| Target | Diagnosis (Benign / Malignant) |

**Dataset shape:** 212,691 rows, 25 columns (after preprocessing)

---

## Methodology

### Models Compared

| Model Type | Variants Tested |
|------------|-----------------|
| Logistic Regression | Plain L1, RBF kernel, Sigmoid kernel |
| Support Vector Machine | Linear, RBF kernel, Polynomial kernel |
| Decision Tree | Pruned with hyperparameter tuning |
| Random Forest | 300 estimators, balanced class weights |
| Neural Network | 3-layer architecture (128 → 64 → 1) with sigmoid activation |

### Preprocessing Steps

1. Dropped irrelevant columns (Patient ID, Country)
2. One-hot encoded categorical variables (Gender, Ethnicity)
3. Label encoded the target variable (Diagnosis)
4. Standardized numeric features using StandardScaler
5. Train/validation/test splits (varied by model for computational efficiency)

### Evaluation Metrics

- **Accuracy** – Overall correct predictions
- **Recall (for malignant class)** – Ability to identify actual cancerous tumors
- **Log Loss** – Confidence calibration of probability predictions
- **Confusion Matrix** – False negatives (malignant predicted benign) prioritized for minimization

---

## Key Findings

| Model | Accuracy | Recall (Malignant) | Log Loss | Notes |
|-------|----------|-------------------|----------|-------|
| Logistic Regression (L1) | 76% | 0.35 | 0.556 | Baseline; poor recall |
| Logistic Regression + RBF | 81% | 0.32 | 0.480 | Better accuracy, worse recall |
| Logistic Regression (with Cancer Risk) | 82% | 0.45 | 0.514 | Adding risk feature helped significantly |
| Linear SVM | 73% | **0.54** | N/A | Best recall of all models |
| RBF SVM | 83% | 0.45 | N/A | Solid all-around performance |
| Decision Tree (pruned) | 83% | 0.45 | N/A | Similar ceiling |
| Random Forest | 83% | 0.46 | N/A | Slightly better recall than RBF SVM |
| Neural Network | 83% | 0.45 | 0.461 | Converged reliably; no breakthrough |

### Performance Ceiling Observation

Every model plateaued at approximately **82–83% accuracy** and **45% recall** for malignant tumors, regardless of complexity. This held across:
- Train, validation, and test sets
- Multiple random seeds (neural network tested with seeds 1–10)
- Hyperparameter tuning

**Diagnosis:** There appears to be inherent overlap in the high-dimensional feature space where malignant and benign cases are not cleanly separable with the available features. A medical dataset with more predictive biomarkers may be required to improve further.

### Feature Importance

When the "Cancer Risk" score was included (initially excluded to test raw predictive power), it dominated all other features:

![Feature Importance](output_27_0.png)

Cancer Risk had near-1 importance — dwarfing all other variables. This suggests either:
1. The risk score is a highly engineered composite feature, or
2. It leaks information about the diagnosis

---

## Real-World Focus: Minimizing False Negatives

In medical diagnosis, **classifying a malignant tumor as benign (false negative)** is far more harmful than the reverse. Throughout the project, I prioritized:

- Adjusting classification thresholds (neural network used 0.375 instead of 0.5 to favor malignant predictions)
- Using `class_weight='balanced'` in SVM models
- Tracking and reporting recall for the malignant class alongside accuracy

Despite these efforts, recall remained around 45% — meaning more than half of cancerous tumors would be missed in practice. **This model is not ready for clinical deployment**, but the analysis identifies clear limitations of the dataset.

---

## Repository Contents

| File | Description |
|------|-------------|
| `thyroid_cancer_analysis.ipynb` | Full Jupyter notebook with all code and outputs |
| `data/Cleaned_thyroidcancer_data.csv` | Preprocessed dataset |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/danieljporter34/thyroid-cancer-classification.git
