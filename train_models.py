"""
Trains the segmentation (KMeans) and retention (XGBoost) models and saves them
as .pkl files for the Streamlit app to load. Run once locally before deploying.
"""
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------- Load & clean ----------
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# ---------- RFM ----------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()

# ---------- Segmentation (KMeans) ----------
rfm_scaler = StandardScaler()
rfm_scaled = rfm_scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Name clusters by their actual behavior (highest Monetary = VIP, etc.)
cluster_stats = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
order = cluster_stats.sort_values("Monetary", ascending=False).index.tolist()
cluster_names = {
    order[0]: "VIP Customers",
    order[1]: "Loyal Customers",
    order[2]: "At-Risk Customers",
    order[3]: "New / Low-Value Customers",
}
rfm["Segment"] = rfm["Cluster"].map(cluster_names)

# ---------- Retention label ----------
customer_dates = df.groupby("CustomerID").agg(
    FirstPurchase=("InvoiceDate", "min"),
    LastPurchase=("InvoiceDate", "max"),
)
customer_dates["DaysActive"] = (customer_dates["LastPurchase"] - customer_dates["FirstPurchase"]).dt.days
customer_dates["Retained"] = (customer_dates["DaysActive"] >= 90).astype(int)

rfm = rfm.merge(customer_dates["Retained"], left_on="CustomerID", right_index=True)

# ---------- Retention model (XGBoost) ----------
X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Retained"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=42,
    eval_metric="logloss",
)
xgb.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score
acc = accuracy_score(y_test, xgb.predict(X_test))
auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"Retention model — Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")

# ---------- Save artifacts ----------
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(rfm_scaler, "scaler.pkl")
joblib.dump(xgb, "retention_model.pkl")
joblib.dump(cluster_names, "cluster_names.pkl")
rfm.to_csv("rfm_table.csv", index=False)  # small, used for the app's overview chart

print("Saved: kmeans_model.pkl, scaler.pkl, retention_model.pkl, cluster_names.pkl, rfm_table.csv")
print("RFM table shape:", rfm.shape)
