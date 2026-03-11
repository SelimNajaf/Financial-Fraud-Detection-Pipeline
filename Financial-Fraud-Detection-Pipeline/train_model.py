"""
Fraud Detection Model Training Pipeline
This script processes financial transaction logs, handles severe class imbalance 
using algorithm-specific class weights (class_weight & scale_pos_weight), trains 
Logistic Regression and XGBoost models via GridSearch, and evaluates them 
using custom decision thresholds.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ==========================================
# 1. DATA LOADING & EXPLORATION
# ==========================================
FILE_PATH = 'train_df.csv'

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: Dataset '{FILE_PATH}' not found. Please ensure it is in the correct directory.")
    sys.exit(1)

print("--- DataFrame Head ---")
print(df.head())

print("\n--- Data Quality Check ---")
print("Missing values per column:\n", df.isnull().sum())

print('\n--- Information ---')
df.info()

# Display the class imbalance ratio
fraud_ratio = df['isFraud'].value_counts(normalize=True) * 100
print(f"\n--- Class Distribution (Fraud Ratio) ---\n{fraud_ratio}\n")

# Drop non-predictive or redundant columns
columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

numeric_columns =['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Generate EDA Plots (Execution pauses until windows are closed)
print("\nGenerating Correlation Heatmap. (Note: Close the plot window to continue...)")
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Numeric Features Correlation")
plt.show()

print("Generating Feature Histograms. (Note: Close the plot window to continue...)")
df[numeric_columns].hist(bins=10, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.show()

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates custom financial features to expose anomalous transaction patterns."""
    data = data.copy()

    # Balance differences
    data["org_balance_diff"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["dest_balance_diff"] = data["newbalanceDest"] - data["oldbalanceDest"]

    # Mathematical inconsistencies in the transaction logic
    data["orig_balance_error"] = data["oldbalanceOrg"] - data["amount"] - data["newbalanceOrig"]
    data["dest_balance_error"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]

    # Binary flags for anomalies
    data["orig_error_flag"] = (data["orig_balance_error"] != 0).astype(int)
    data["dest_error_flag"] = (data["dest_balance_error"] != 0).astype(int)

    # Flag for accounts that remain entirely empty
    data["dest_balance_empty"] = ((data["oldbalanceDest"] == 0) & (data["newbalanceDest"] == 0)).astype(int)

    # Proportion of the amount transferred relative to the original balance
    data["amount_org_ratio"] = data["amount"] / (data["oldbalanceOrg"] + 1)
    data["amount_org_ratio"] = data["amount_org_ratio"].clip(upper=1e5)

    return data

df = add_features(df)

# ==========================================
# 3. DATA SPLITTING & PREPROCESSING
# ==========================================
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Stratified split ensures the train and test sets have the same proportion of fraud cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Categorize columns based on their data types
categorical_columns = ['type']
numeric_columns = [col for col in X.columns if col not in categorical_columns]

# Set up the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
    ]
)

# ==========================================
# 4. MODEL CONFIGURATION & TRAINING
# ==========================================
# Calculate the ratio of negative cases to positive cases
# This dynamically informs XGBoost to pay proportionately more attention to the minority (Fraud) class
negative_cases = (y_train == 0).sum()
positive_cases = (y_train == 1).sum()
scale_weight = negative_cases / positive_cases

print(f"Non-fraud to fraud ratio (scale_pos_weight): {scale_weight:.2f}")

# Define the models with built-in imbalance handling
models = {
    # 'balanced' automatically adjusts weights inversely proportional to class frequencies
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    
    # 'scale_pos_weight' applies the exact ratio calculated above
    "XGBoost": XGBClassifier(eval_metric="logloss", tree_method="hist", scale_pos_weight=scale_weight, random_state=42)
}

# Simplify the parameter grids to mitigate overfitting risks
params = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1]
    },
    "XGBoost": {
        "model__n_estimators":[100, 200],
        "model__max_depth": [3, 4],
        "model__learning_rate": [0.05, 0.1]
    }
}

best_score = 0
trained_model = None
best_model_name = None

for model_name, model in models.items():
    print(f'\n--- Training {model_name} ---')

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        params[model_name],
        cv=3, 
        scoring="f1",
        n_jobs=-1
    )

    # Train the pipeline
    grid.fit(X_train, y_train)
    current_best_model = grid.best_estimator_

    # Evaluate on the test dataset
    y_pred = current_best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"{model_name} Best Parameters: {grid.best_params_}")
    print(f"{model_name} Test F1 Score: {f1:.4f}")

    # Keep track of the highest performing model
    if f1 > best_score:
        best_score = f1
        trained_model = current_best_model
        best_model_name = model_name

print("\n" + "="*45)
print(f"Winning Model: {best_model_name}")
print(f"Best Test F1 Score: {best_score:.4f}")
print("="*45)


# ==========================================
# 5. MODEL EVALUATION
# ==========================================
print("\n" + "="*45)
print("  EVALUATION 1: STANDARD THRESHOLD (0.50)")
print("="*45)
y_pred_standard = trained_model.predict(X_test)
print(classification_report(y_test, y_pred_standard))


print("="*45)
print("  EVALUATION 2: STRICT THRESHOLD (0.90)")
print("="*45)
# Extract probabilities for the positive class (Fraud = 1)
y_pred_probabilities = trained_model.predict_proba(X_test)[:, 1]

# Apply a strict custom threshold to prioritize precision and minimize false alarms
CUSTOM_THRESHOLD = 0.90
y_pred_strict = (y_pred_probabilities >= CUSTOM_THRESHOLD).astype(int)

print(classification_report(y_test, y_pred_strict))


# ==========================================
# 6. INFERENCE SIMULATION (TEST TRANSACTIONS)
# ==========================================
transactions = {
    # ================= SAFE =================
    "SAFE_1": {
        'step': 3, 
        'type': 'PAYMENT', 
        'amount': 342, 
        'oldbalanceOrg': 5200, 
        'newbalanceOrig': 4858, 
        'oldbalanceDest': 2100, 
        'newbalanceDest': 2442
    },
    "SAFE_2": {
        'step': 25, 
        'type': 'TRANSFER', 
        'amount': 4100, 
        'oldbalanceOrg': 15000, 
        'newbalanceOrig': 10850, 
        'oldbalanceDest': 5000, 
        'newbalanceDest': 9100
    },

    # ================= FRAUD =================
    "FRAUD_REAL_1": {
        'step': 45, 
        'type': 'TRANSFER', 
        'amount': 95000.0, 
        'oldbalanceOrg': 95000.0, 
        'newbalanceOrig': 0.0, 
        'oldbalanceDest': 0.0, 
        'newbalanceDest': 0.0
    },
    "FRAUD_REAL_2": {
        'step': 46, 
        'type': 'CASH_OUT', 
        'amount': 95000.0, 
        'oldbalanceOrg': 95000.0, 
        'newbalanceOrig': 0.0, 
        'oldbalanceDest': 21000.0, 
        'newbalanceDest': 116000.0 
    },
    "FRAUD_REAL_3": {
        'step': 50, 
        'type': 'TRANSFER', 
        'amount': 1500000.0, 
        'oldbalanceOrg': 1500000.0, 
        'newbalanceOrig': 0.0, 
        'oldbalanceDest': 0.0, 
        'newbalanceDest': 0.0
    }
}

def predict_transaction(data: dict):
    """Simulates an API request applying feature engineering and returning the prediction."""
    df_test = pd.DataFrame([data])
    df_test = add_features(df_test)
    
    # Ensure correct column ordering based on training set
    df_test = df_test[X_train.columns]

    prob = trained_model.predict_proba(df_test)[:,1][0]
    pred = trained_model.predict(df_test)[0]

    return prob, pred

print("\n--- TEST RESULTS ---")
for label, data in transactions.items():
    prob, pred = predict_transaction(data)
    
    # Clean up extremely small microscopic probabilities (e.g., 1e-5) for display purposes
    prob_clean = 0.0 if prob < 0.001 else prob
    
    result = "FRAUD" if pred == 1 else "SAFE"
    print(f"{label:<15} -> {result:<10} | Probability: {prob_clean:.4f}")


# ==========================================
# 7. EXPORT MODEL
# ==========================================
MODEL_FILENAME = 'trained_model.joblib'
joblib.dump(trained_model, MODEL_FILENAME)
print(f"\nTraining complete! Model successfully saved as '{MODEL_FILENAME}'")