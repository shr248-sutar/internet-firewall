import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model

# Page configuration
st.set_page_config(page_title="ML Action Classifier", layout="wide")

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset Summary", "Predict"])

# Cache model loading
@st.cache_resource
def load_model():
    model, accuracy = train_model()
    return model, accuracy

model, accuracy = load_model()

# Cache dataset loading
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/2035tu/Desktop/streamlit/log2_trimmed.csv")
    df = df[df['Action'].isin(['allow', 'deny'])]
    return df

df = load_data()

# -------------------------------
# 1. HOME PAGE
# -------------------------------
if page == "Home":
    st.title("🚦 Network Action Classifier")
    st.write("""
        Welcome to the **Network Action Classifier** using Machine Learning!  
        This tool predicts whether a network action should be **allowed** or **denied** based on firewall log data.
        
        Use the sidebar to:
        - Explore the dataset
        - Predict network actions using input features
    """)

    st.metric("🔍 Model Accuracy", f"{accuracy * 100:.2f}%")

# -------------------------------
# 2. DATASET SUMMARY PAGE
# -------------------------------
elif page == "Dataset Summary":
    st.title("📊 Dataset Summary")
    st.write("A quick overview of the Internet Firewall dataset used for training.")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔄 Action Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Action", ax=ax1)
    st.pyplot(fig1)

    st.subheader("📦 Feature Correlation Heatmap")
    df_encoded = df.copy()
    df_encoded["Action"] = df_encoded["Action"].map({"allow": 0, "deny": 1})
    corr = df_encoded.corr(numeric_only=True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -------------------------------
# 3. PREDICT PAGE
# -------------------------------
elif page == "Predict":
    st.title("🔮 Predict Network Action")
    st.subheader("Enter Network Log Features to Predict")

    with st.form("prediction_form"):
        src_port = st.number_input("Source Port", min_value=0)
        dst_port = st.number_input("Destination Port", min_value=0)
        nat_src_port = st.number_input("NAT Source Port", min_value=0)
        nat_dst_port = st.number_input("NAT Destination Port", min_value=0)
        bytes_all = st.number_input("Bytes", min_value=0)
        bytes_sent = st.number_input("Bytes Sent", min_value=0)
        bytes_received = st.number_input("Bytes Received", min_value=0)
        packets = st.number_input("Packets", min_value=0)
        elapsed = st.number_input("Elapsed Time (sec)", min_value=0)
        pkts_sent = st.number_input("Packets Sent", min_value=0)
        pkts_received = st.number_input("Packets Received", min_value=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[
            src_port, dst_port, nat_src_port, nat_dst_port,
            bytes_all, bytes_sent, bytes_received,
            packets, elapsed, pkts_sent, pkts_received
        ]], columns=[
            "Source Port", "Destination Port", "NAT Source Port", "NAT Destination Port",
            "Bytes", "Bytes Sent", "Bytes Received",
            "Packets", "Elapsed Time (sec)", "pkts_sent", "pkts_received"
        ])

        prediction = model.predict(input_data)[0]
        result = "✅ Allow" if prediction == 0 else "⛔ Deny"

        st.subheader("🧠 Prediction Result")
        st.info(f"The model predicts: **{result}**")
