import streamlit as st
import pandas as pd
from model import train_model

st.set_page_config(page_title="ML Action Classifier", layout="centered")

st.title("🚦 Network Action Classifier")
st.write("This ML model predicts whether a network action is **allow** or **deny** based on network log data.")

# Train model and get accuracy
with st.spinner("Training model..."):
    model, accuracy = train_model()
st.success("Model trained successfully!")

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.subheader("🔢 Input Network Features")

# Input form
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

# Handle prediction
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

    st.subheader("Prediction Result")
    st.info(f"The model predicts: **{result}**")
