"""
Fraud Detection API
A FastAPI web service that loads a pre-trained machine learning model 
to identify fraudulent transactions dynamically using custom feature engineering.
"""

import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================
# 1. MODEL CONFIGURATION & LOADING
# ==========================================
MODEL_PATH = 'trained_model.joblib'

try:
    # Load the pre-trained machine learning pipeline into memory upon startup
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please ensure you run the model training script before starting the API.")
    sys.exit(1)

# Ensure the columns strictly match the exact order expected by the trained model
EXPECTED_COLUMNS =[
    'step', 
    'type', 
    'amount', 
    'oldbalanceOrg', 
    'newbalanceOrig',
    'oldbalanceDest', 
    'newbalanceDest', 
    'org_balance_diff',
    'dest_balance_diff', 
    'orig_balance_error', 
    'dest_balance_error',
    'orig_error_flag', 
    'dest_error_flag', 
    'dest_balance_empty',
    'amount_org_ratio'
]

# Hardcoded strict business limit for flagging fraud
FRAUD_THRESHOLD = 0.90


# ==========================================
# 2. FEATURE ENGINEERING HELPER
# ==========================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the exact mathematical transformations used during model training."""
    df = df.copy()

    # Balance differences
    df["org_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Mathematical inconsistencies
    df["orig_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Binary anomaly flags
    df["orig_error_flag"] = (df["orig_balance_error"] != 0).astype(int)
    df["dest_error_flag"] = (df["dest_balance_error"] != 0).astype(int)

    # Empty account flags
    df["dest_balance_empty"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)

    # Transfer ratio
    df["amount_org_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_org_ratio"] = df["amount_org_ratio"].clip(upper=1e5)

    return df


# ==========================================
# 3. FASTAPI INITIALIZATION & SCHEMAS
# ==========================================
app = FastAPI(
    title="Fraud Detection API",
    description="An API that evaluates financial transactions in real-time to detect fraudulent behavior.",
    version="1.0.0"
)

class TransactionData(BaseModel):
    """Schema defining the expected JSON payload for incoming transactions."""
    step: float = Field(..., description="Maps a unit of time in the real world (1 step is 1 hour)")
    type: str = Field(..., description="Transaction type (e.g., 'CASH_OUT', 'PAYMENT', 'TRANSFER')")
    amount: float = Field(..., description="Amount of the transaction in local currency")
    oldbalanceOrg: float = Field(..., description="Initial balance before the transaction")
    newbalanceOrig: float = Field(..., description="New balance after the transaction")
    oldbalanceDest: float = Field(..., description="Initial balance of recipient before transaction")
    newbalanceDest: float = Field(..., description="New balance of recipient after transaction")


# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.post('/predict')
async def predict_fraud(data: TransactionData) -> dict:
    """
    Receives financial transaction data, engineers advanced features, 
    and predicts whether the transaction is safe or fraudulent.
    """
    try:
        # 1. Convert the validated Pydantic object into a pandas DataFrame
        df = pd.DataFrame([data.model_dump()])
        
        # 2. Append the engineered features
        df = add_features(df)
        
        # 3. Reorder the columns to perfectly match the training environment's order
        df = df[EXPECTED_COLUMNS]

        # 4. Extract the probability of the positive class (Fraud = 1)
        prob = float(model.predict_proba(df)[:,1][0])

        # Clean up microscopically small probabilities for UI aesthetics
        prob_clean = 0.0 if prob < 0.001 else prob

        # 5. Apply the custom business logic threshold
        result = "FRAUD" if prob_clean >= FRAUD_THRESHOLD else "SAFE"

        # Return structured decision
        return {
            "prediction": result,
            "fraud_probability": round(prob_clean, 4)  # Rounding makes the JSON payload cleaner
        }
        
    except Exception as e:
        # Gracefully handle and report any unexpected errors
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {str(e)}")