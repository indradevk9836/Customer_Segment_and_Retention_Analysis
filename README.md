# Customer Segmentation & Retention Analysis

RFM-based customer segmentation (K-Means) and churn/retention prediction (XGBoost) on the
[Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

## What this project does
- Cleans raw transaction-level e-commerce data
- Engineers RFM (Recency, Frequency, Monetary) features per customer
- Segments customers into groups (VIP, Loyal, At-Risk, New) using K-Means clustering
- Builds a retention/churn prediction model (Logistic Regression, Random Forest, XGBoost — compared via
  accuracy, precision, recall, F1, ROC-AUC and 5-fold cross-validation)
- Tunes the final XGBoost model with GridSearchCV / RandomizedSearchCV
- Outputs a per-customer report: segment, retention status, and retention probability

## Setup
```bash
pip install -r requirements.txt
```

Download `OnlineRetail.csv` from the link above and place it in the project root
(excluded from this repo — see `.gitignore`).

## Run
Open `Customer_Segmentation___Retention_Analysis.ipynb` in Jupyter and run all cells top to bottom.

## Tech
Python, pandas, scikit-learn, XGBoost, Plotly
