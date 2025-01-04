# Breast Cancer Diagnosis Using Machine Learning

This project uses machine learning models to diagnose breast cancer based on the Breast Cancer Wisconsin (Diagnostic) dataset. The models implemented include Logistic Regression and Random Forest, with feature selection and hyperparameter tuning for improved performance.

---

## Features

- **Exploratory Data Analysis (EDA):** Visualize data trends and relationships.
- **Data Preprocessing:**
  - Handling missing values.
  - Feature scaling using `StandardScaler`.
- **Machine Learning Models:**
  - Logistic Regression.
  - Random Forest Classifier.
- **Hyperparameter Tuning:** Grid search for optimizing Logistic Regression.
- **Model Evaluation:**
  - Accuracy, Confusion Matrix, and Classification Report.
  - Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC).
- **Feature Selection:** Recursive Feature Elimination (RFE) for feature importance analysis.
- **Model Persistence:** Save trained models and scalers using `joblib`.

---

## Installation

### Prerequisites
- Python 3.7+
- Required libraries listed in `requirements.txt`.

### Steps
1. Download the dataset (if not included) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29).

---


## Project Structure

```plaintext
breast-cancer-diagnosis/
├── breast_cancer_diagnosis.py    # Main script
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── saved_models/                 # Directory for saved models
