from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

model = pickle.load(open("xgboost_churn_model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(
    SeniorCitizen: int,
    tenure: int,
    MonthlyCharges: float,
    TotalCharges: float
):
    input_data = pd.DataFrame({
        "SeniorCitizen": [SeniorCitizen],
        "tenure": [tenure],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]

    result = "Churn" if prediction == 1 else "Stay"

    return {"prediction": result}