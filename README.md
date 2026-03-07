# 🚨 Financial Fraud Detection Pipeline

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-brightgreen?style=for-the-badge)
![FinTech](https://img.shields.io/badge/Domain-FinTech_%7C_Fraud-red?style=for-the-badge)

## 📖 Project Overview
The **Financial Fraud Detection Pipeline** is a machine learning solution designed to identify fraudulent financial transactions in highly imbalanced datasets. In real-world financial systems, fraudulent transactions represent a microscopic fraction of total volume (in this dataset: **0.13%**). Standard machine learning models fail under these conditions by simply predicting all transactions as "safe."

This project solves this severe class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** wrapped securely within an `imblearn` pipeline to prevent data leakage. Furthermore, it demonstrates production-level decision-making by tuning the classification threshold to mathematically balance the trade-off between catching fraud (Recall) and preventing false alarms for legitimate users (Precision).

## ✨ Key Features
*   **Imbalanced Data Handling (SMOTE):** Synthetically generates new examples of the minority class (fraud) during the training phase to force the model to learn fraudulent patterns rather than ignoring them.
*   **Leakage-Proof Pipeline:** Utilizes `imblearn.pipeline.Pipeline` instead of the standard `sklearn` pipeline. This ensures SMOTE is *only* applied to the training folds during cross-validation, preventing catastrophic data leakage into the test set.
*   **Custom Decision Thresholds:** Shifts the decision boundary from the default `0.50` to a strict `0.95` to drastically reduce False Positives (customer friction) while maintaining a strong detection rate.
*   **API Inference Simulation:** Includes a robust evaluation script that simulates how a production API would receive a JSON payload, calculate the fraud probability, and block or approve the transaction in real time.

## 📊 Data Description
The model is trained on financial transaction logs (`PS_20174392719_1491204439457_log.csv`). 
*🔗 **Dataset Link:**[Insert Dataset Link Here - e.g., Kaggle PaySim Dataset]*

**Class Distribution:**
*   ✅ **Safe (0):** 99.87%
*   🚨 **Fraud (1):** 0.13%

**Input Features:**
*   `step`: Maps a unit of time in the real world (1 step = 1 hour).
*   `type`: Type of transaction (e.g., TRANSFER, CASH_OUT).
*   `amount`: Transaction amount in local currency.
*   `oldbalanceOrg` / `newbalanceOrig`: Origin account balances before and after the transaction.
*   `oldbalanceDest` / `newbalanceDest`: Destination account balances before and after the transaction.

*(Note: High-cardinality nominal features like `nameOrig` and `nameDest` were dropped to prevent overfitting).*

## 🛠️ Project Architecture

```text
├── train_model.ipynb      # Main script: Data prep, SMOTE, Training, and Evaluation
├── train_df.csv           # Raw transaction dataset [Not included, download required]
└── README.md              # Project documentation
```

## 🚀 Installation & Prerequisites

To run this project locally, ensure you have Python 3.8+ installed. 

1. **Clone the repository:**
   ```bash
   git clone[Insert Repository Link Here]
   cd [Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install pandas scikit-learn imbalanced-learn
   ```

3. **Add the Dataset:**
   Ensure the dataset `.csv` file is downloaded and placed in the root directory of the project before running the training script.

## 💻 Usage / How to Run

Execute the main pipeline script. The script will automatically check data quality, apply stratifications, train the SMOTE pipeline, output threshold evaluations, and simulate an API transaction.

```bash
python fraud_detection_pipeline.py
```

## 📈 Results & Threshold Tuning

When evaluating highly imbalanced data, Accuracy is a misleading metric (predicting "0" for everything yields 99.8% accuracy). Instead, we focus on Precision, Recall, and the F1-Score.

### Evaluation 1: Standard Threshold (0.50)
*   **Recall:** `96%` (Catches almost all fraud)
*   **Precision:** `2%` (Massive amount of false alarms)
*   *Business Impact:* Too many legitimate customers would have their credit cards blocked, leading to severe customer churn.

### Evaluation 2: Strict Threshold (0.97)
*   **Recall:** `68%` (Still catches the vast majority of fraud)
*   **Precision:** `20%` (Significant improvement in prediction quality)
*   **Accuracy:** `100%`
*   *Business Impact:* By requiring the model to be 97% confident before flagging a transaction, we successfully reduce operational overhead and false blocks, achieving a much more viable production model.

### API Inference Output
The script concludes with a simulated production payload:
> `--- API Inference Simulation ---`
> `Fraud probability for this transaction: 81.04%`
> `Decision: ✅ SAFE (Transaction Approved)`
*(Because 81.04% is below our strict 95% custom threshold, the transaction is allowed).*

## 🤝 Contributing
Contributions are highly encouraged! If you'd like to improve the model (e.g., implementing XGBoost/Random Forest instead of Logistic Regression, or deploying this as a Flask/FastAPI microservice):
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/FastAPIDeployment`)
3. Commit your Changes (`git commit -m 'Deploy SMOTE model via FastAPI'`)
4. Push to the Branch (`git push origin feature/FastAPIDeployment`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf-data-analyst](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
