import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from ucimlrepo import fetch_ucirepo
import urllib.request
import ssl
import certifi
import time

# Function to retry download with error handling
def fetch_data_with_retry(id, max_retries=3, retry_delay=5):
    """Fetches data from UCI repository with retries and error handling.

    Args:
        id: The ID of the dataset to fetch.
        max_retries: The maximum number of retries.
        retry_delay: The delay in seconds between retries.

    Returns:
        The fetched dataset.

    Raises:
        ConnectionError: If the connection fails after multiple retries.
    """
    for attempt in range(max_retries):
        try:
            return fetch_ucirepo(id=id)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:  # Only sleep if there are more retries
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise ConnectionError("Error connecting to server after multiple retries")
        except ConnectionError as e:
            print(f"Connection error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise

# Load the dataset with retry logic
breast_cancer_wisconsin_diagnostic = fetch_data_with_retry(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Print the first few rows of the data
print(X.head())
print(y.head())

# Exploratory Data Analysis (EDA)
sns.pairplot(pd.concat([X, y], axis=1), hue="Diagnosis") # Change hue to "Diagnosis"
plt.show()

# Data Preprocessing: Handling Missing Values and Feature Scaling
# Check for missing values
print(X.isnull().sum())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Model Selection: Logistic Regression and Random Forest
# Logistic Regression with class weight adjustment
log_reg_model = LogisticRegression(class_weight='balanced')
log_reg_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train, y_train)

# Cross-validation for model evaluation
log_reg_cv_scores = cross_val_score(log_reg_model, X_scaled, y, cv=5)
rf_cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)
print("Logistic Regression CV scores:", log_reg_cv_scores)
print("Random Forest CV scores:", rf_cv_scores)

# Hyperparameter Tuning: Grid Search for Logistic Regression
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", grid_search.best_params_)

# Evaluate Logistic Regression Model
y_pred_log_reg = log_reg_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Evaluate Random Forest Model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))