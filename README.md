# Pneumonia Detection ML Pipeline

This project tests seven classification models on a pneumonia dataset, compares their performance using cross-validation and hyperparameter tuning, and combines the strongest models into a voting ensemble. The full pipeline covers data cleaning, feature preparation, model training, evaluation, and ensemble construction with resaoning provided for each step.

---

## Dataset

The dataset (`pneumonia_raw.csv`) contains data manually extracted from X-ray scans and will be used to predict pneumonia based on the data.

**Data cleaning steps applied:**
- Removed duplicates and rows with missing values
- Dropped the `Patient_ID` column (not predictive)
- Removed invalid data
- Converted `yes/no` diagnosis strings to binary `1/0`

---

## Models Evaluated

| Model | Cross-Val AUC |
|---|---|
| Decision Tree | 71.1% |
| Random Forest | 74.1% |
| K-Nearest Neighbours | 67.9% |
| Naive Bayes | 73.1% |
| SGD Classifier | 70.2% |
| SVM | 68.6% |
| MLP (Artificial Neural Network) | 74.5% |
| **Voting Ensemble (best models)** | 75.0% |
| Bagging Ensemble | 72.7% |
| Random Subspaces Ensemble | 74.4% |

All models were trained inside scikit-learn `Pipeline` objects with `StandardScaler` preprocessing. Hyperparameters were tuned using `GridSearchCV` or manual loop search, evaluated with `StratifiedKFold` (5 folds).

---

## Results

The best model was the voting ensemble which achieved:

- **ROC-AUC: 0.75**
- **Precision: 0.92** (positive class)

Evaluation metrics used: ROC-AUC, precision, recall, F1-score, confusion matrix, ROC curve plots.

---

## Environment

This notebook was developed in **Google Colab** with data loaded from Google Drive. To run it locally:

1. Remove the `drive.mount(...)` cell
2. Update the file path in `pd.read_csv(...)` to point to your local copy of the dataset

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
