import streamlit as st
import requests

st.title("Customer Churn Prediction App")

# Your Render API URL
API_URL = "https://customer-churn-prediction-3lal.onrender.com/predict"

# Inputs
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0, 200, 50)
total = st.slider("Total Charges", 0, 10000, 500)

if st.button("Predict"):

    params = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    response = requests.post(API_URL, params=params)

    if response.status_code == 200:
        result = response.json()["prediction"]

        if result == "Churn":
            st.error("Customer will churn")
        else:
            st.success("Customer will stay")
    else:
        st.error("API error. Please try again.")