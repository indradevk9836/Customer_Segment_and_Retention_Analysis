import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation & Retention", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    retention_model = joblib.load("retention_model.pkl")
    cluster_names = joblib.load("cluster_names.pkl")
    rfm = pd.read_csv("rfm_table.csv")
    return kmeans, scaler, retention_model, cluster_names, rfm

kmeans, scaler, retention_model, cluster_names, rfm = load_artifacts()

st.title("🛍️ Customer Segmentation & Retention Analysis")
st.caption("RFM segmentation (K-Means) + churn/retention prediction (XGBoost) on the Online Retail dataset")

tab1, tab2 = st.tabs(["🔍 Predict a Customer", "📊 Segment Overview"])

# ---------- TAB 1: single customer prediction ----------
with tab1:
    st.subheader("Enter customer RFM values")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5)
    with col3:
        monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, value=500.0, step=50.0)

    if st.button("Analyze Customer", type="primary"):
        input_df = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})

        # Segment
        scaled = scaler.transform(input_df)
        cluster = kmeans.predict(scaled)[0]
        segment = cluster_names[cluster]

        # Retention
        retained = retention_model.predict(input_df)[0]
        retain_prob = retention_model.predict_proba(input_df)[0][1]

        st.markdown("### Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Segment", segment)
        r2.metric("Retention Status", "Likely Retained" if retained == 1 else "Likely to Churn")
        r3.metric("Retention Probability", f"{retain_prob:.1%}")

        st.markdown("### Recommended Action")
        if segment == "VIP Customers" and retain_prob < 0.5:
            st.warning("🚨 High-value customer at churn risk — prioritize for a retention offer immediately.")
        elif segment == "VIP Customers":
            st.success("✅ High-value, likely to stay — good candidate for early access to new features / loyalty perks.")
        elif retain_prob < 0.4:
            st.info("⚠️ Low retention probability and lower value — deprioritize for costly interventions; consider low-cost automated re-engagement (email) instead.")
        else:
            st.info("🙂 Stable customer — monitor, no urgent action needed.")

# ---------- TAB 2: overview ----------
with tab2:
    st.subheader("Customer base overview")
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(seg_counts, hide_index=True, use_container_width=True)
    with c2:
        fig_pie = px.pie(seg_counts, names="Segment", values="Customers", title="Segment Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    fig_scatter = px.scatter(
        rfm, x="Frequency", y="Monetary", color="Segment",
        hover_data=["Recency"], title="Customers by Frequency vs Monetary Value",
        log_y=True,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    retention_by_seg = rfm.groupby("Segment")["Retained"].mean().reset_index()
    retention_by_seg["Retained"] = (retention_by_seg["Retained"] * 100).round(1)
    fig_bar = px.bar(
        retention_by_seg, x="Segment", y="Retained",
        title="Retention Rate by Segment (%)", text="Retained",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
