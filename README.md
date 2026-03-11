# 🚨 Advanced Financial Fraud Detection Engine & API

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-orange?style=for-the-badge&logo=xgboost)
![Imbalanced Data](https://img.shields.io/badge/Class_Imbalance-0.13%25_Fraud-red?style=for-the-badge)

## 📖 Project Overview
The **Advanced Financial Fraud Detection Engine** is a production-grade machine learning pipeline and API designed to detect illicit transactions in real-time. Financial fraud datasets are notoriously imbalanced (in this project, fraud accounts for just **0.13%** of the data). This solution tackles this challenge not through simple resampling, but through mathematical feature engineering and algorithmic penalization using **XGBoost**.

This repository contains the complete end-to-end lifecycle: automated Exploratory Data Analysis (EDA), business-logic feature engineering, hyperparameter tuning via `GridSearchCV`, threshold optimization, and an asynchronous REST API built with **FastAPI** that evaluates live transactions in milliseconds.

## ✨ Key Features
*   **Strategic Feature Engineering:** The model doesn't just look at raw numbers; it mathematically calculates account balance discrepancies (`orig_balance_error`, `dest_balance_error`) to expose the hidden mechanics of fraudulent transfers.
*   **Dynamic Imbalance Handling:** Automatically calculates the negative-to-positive class ratio (`773.75:1`) and injects it into XGBoost's `scale_pos_weight` parameter, forcing the algorithm to aggressively penalize missed fraud cases without data leakage.
*   **Threshold Tuning for Production:** Adjusts the default probability decision boundary from `0.50` to a strict `0.90`. This eliminates "False Positives" (preventing legitimate users from having their cards falsely blocked) while maintaining a near-perfect recall rate.
*   **FastAPI Microservice:** The model is wrapped in a high-performance, asynchronous web service. It perfectly mirrors the training environment by dynamically engineering features from raw JSON payloads on the fly.
*   **Strict Pydantic Validation:** Ensures all incoming API traffic is strictly typed and formatted, preventing server crashes from malformed requests.

## 📊 Data Description
The model is trained on a synthetic dataset of over 6.3 million financial transaction logs (`train_df.csv`).

**Input Features:**
*   `step`: Maps a unit of time in the real world (1 step = 1 hour).
*   `type`: Transaction type (e.g., `TRANSFER`, `CASH_OUT`, `PAYMENT`).
*   `amount`: Transaction amount in local currency.
*   `oldbalanceOrg` / `newbalanceOrig`: Origin account balances before and after the transaction.
*   `oldbalanceDest` / `newbalanceDest`: Destination account balances before and after the transaction.

## 🛠️ Project Architecture

```text
├── train_model.py                             # EDA, Feature Engineering, GridSearch, & Evaluation
├── main.py                                    # FastAPI application & real-time prediction logic
├── trained_model.joblib                       # Serialized XGBoost pipeline (Generated Output)
└── README.md                                  # Project documentation
```

## 🚀 Installation & Prerequisites

To run this pipeline and API locally, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone [Insert Repository Link Here]
   cd [Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn fastapi uvicorn joblib pydantic
   ```

3. **Add the Dataset:**
   Ensure the `train_df.csv` file is downloaded and placed in the root directory.

## 💻 Usage / How to Run

### Step 1: Train the Model
Run the training script to perform EDA, engineer features, tune the algorithms, evaluate thresholds, and export the `.joblib` model. 
*(Note: You will need to close the pop-up EDA plot windows for the script to continue training).*

```bash
python train_model.py
```

### Step 2: Launch the FastAPI Server
Once the model is serialized as `trained_model.joblib`, spin up the API using Uvicorn.

```bash
uvicorn main:app --reload
```

### Step 3: Test the API Endpoint
Navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI, or simulate a live transaction via cURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "step": 45,
  "type": "TRANSFER",
  "amount": 95000.0,
  "oldbalanceOrg": 95000.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}'
```

## 📈 Results / Model Evaluation

By calculating the mathematical errors between the transfer amount and the resulting account balances, the **XGBoost** model was able to successfully "crack" the underlying fraud patterns, entirely outperforming Logistic Regression.

**GridSearch Performance:**
*   **Logistic Regression F1 Score:** `0.0902`
*   **XGBoost F1 Score:** `0.9667` 🏆

**Threshold Tuning (Strict 0.90 Limit):**
At a strict probability threshold of `0.90`, the model achieved near-perfect evaluation metrics on the test set:
*   **Precision:** `0.99` (Virtually zero false alarms for legitimate customers)
*   **Recall:** `1.00` (Caught 100% of the fraud cases in the test set)

**Example API Response (Fraud Blocked):**
```json
{
  "prediction": "FRAUD",
  "fraud_probability": 1.0
}
```

## 🤝 Contributing
Contributions are highly encouraged! To further optimize this project:
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/Dockerization`)
3. Commit your Changes (`git commit -m 'Add Docker support for API'`)
4. Push to the Branch (`git push origin feature/Dockerization`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
