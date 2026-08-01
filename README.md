# 🛍️ Customer-Segment-and-Retention-Analysis

An end-to-end Machine Learning project that performs **Customer Segmentation** and **Customer Retention Prediction** using the Online Retail dataset.

The system groups customers into meaningful business segments using clustering techniques and predicts whether a customer is likely to be retained using a supervised machine learning model.

---

## 📌 Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering using RFM Analysis
- Customer Segmentation using K-Means Clustering
- Customer Retention Prediction using XGBoost
- Hyperparameter Tuning with RandomizedSearchCV
- Model Evaluation and Comparison
- Interactive Streamlit Web Application
- Business Recommendations based on customer segment

---

## 📊 Dataset

**Dataset:** Online Retail Dataset

The dataset contains transactions occurring between **01/12/2010 and 09/12/2011** for a UK-based online retail company.

### Features Used

- CustomerID
- InvoiceDate
- Quantity
- UnitPrice

### Engineered Features

- Recency
- Frequency
- Monetary (RFM)
- Total Price

---

## 🧠 Machine Learning Models

### Customer Segmentation

 Model

- **K-Means Clustering**

---

### Customer Retention Prediction

Algorithms Compared

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Naive Bayes
- XGBoost

Final Model

- **XGBoost**

Hyperparameter tuning was performed using **RandomizedSearchCV**.

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## 🚀 Streamlit Application

The application allows users to:

- Enter customer Recency, Frequency and Monetary values
- Predict Customer Segment
- Predict Customer Retention Status
- Display Retention Probability
- Provide Business Recommendation

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 📈 Results

### Customer Segmentation

The project successfully segmented customers into meaningful groups such as:

- VIP Customers
- Loyal Customers
- New Customers
- At-Risk Customers

### Customer Retention

Among all evaluated models, **XGBoost** achieved the best overall performance and was selected as the final model after hyperparameter tuning.

---

## 📷 Application Preview

Add screenshots of your Streamlit dashboard here.

Example:

```
images/home.png

images/prediction.png
```

---

## 🔮 Future Improvements

- Real-time customer prediction
- Customer Lifetime Value (CLV) Prediction
- Cloud Deployment

---

## 👨‍💻 Author

**Indradev Kumar**
