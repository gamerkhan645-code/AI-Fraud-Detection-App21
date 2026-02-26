import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:bold;
    color:#1f77b4;
}
.metric-box {
    background-color:#111827;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🛡️ AI Financial Fraud Detection Dashboard</p>', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Settings")

contamination = st.sidebar.slider(
    "Fraud Sensitivity",
    0.01, 0.20, 0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

# -------------------- SAMPLE DATA --------------------
def generate_sample_data(n=500):
    np.random.seed(42)
    data = pd.DataFrame({
        "amount": np.random.normal(200, 50, n),
        "oldbalanceOrg": np.random.normal(5000, 2000, n),
        "newbalanceOrig": np.random.normal(4800, 2100, n),
        "oldbalanceDest": np.random.normal(3000, 1500, n),
        "newbalanceDest": np.random.normal(3200, 1600, n),
    })

    fraud_index = np.random.choice(n, size=int(0.05*n), replace=False)
    data.loc[fraud_index, "amount"] *= 6

    return data

# -------------------- LOAD DATA --------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded")
else:
    df = generate_sample_data()
    st.info("ℹ️ Using AI-generated sample financial data")

st.subheader("📊 Transaction Data")
st.dataframe(df.head())

# -------------------- AI MODEL --------------------
# Keep only amount as main fraud indicator
if "amount" in df.columns:
    features = df[["amount"]]
else:
    st.error("CSV must contain an 'amount' column.")
    st.stop()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

model = IsolationForest(
    contamination=contamination,
    random_state=42
)

predictions = model.fit_predict(scaled_data)

df["fraud_prediction"] = predictions
df["fraud_prediction"] = df["fraud_prediction"].map({
    1: "Legitimate",
    -1: "Fraud"
})

# -------------------- METRICS --------------------
total = len(df)
frauds = (df["fraud_prediction"] == "Fraud").sum()
legit = total - frauds

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total)
col2.metric("Fraud Detected", frauds)
col3.metric("Legitimate", legit)

# -------------------- CHART --------------------
st.subheader("📈 Fraud Distribution")

fig = px.pie(
    names=["Fraud", "Legitimate"],
    values=[frauds, legit],
    title="Fraud vs Legitimate Transactions"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------- TABLE --------------------
st.subheader("🚨 Fraud Cases")
st.dataframe(df[df["fraud_prediction"] == "Fraud"].head(20))

# -------------------- DOWNLOAD --------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Results",
    data=csv,
    file_name="fraud_detection_results.csv",
    mime="text/csv"
)