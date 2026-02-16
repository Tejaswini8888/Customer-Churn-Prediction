import streamlit as st
import requests

st.title("Customer Churn Prediction")

senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0, 200, 50)
total = st.slider("Total Charges", 0, 10000, 500)

if st.button("Predict"):

    url = "https://customer-churn-prediction-3lal.onrender.com/predict"

    params = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    response = requests.post(url, params=params)

    result = response.json()["prediction"]

    if result == "Churn":
        st.error("Customer will churn")
    else:
        st.success("Customer will stay")