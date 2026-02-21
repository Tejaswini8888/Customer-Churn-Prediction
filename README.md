# 📊 Customer Churn Prediction API
🚀 XGBoost + FastAPI | Production-Ready ML System

A production-ready Machine Learning system that predicts customer churn using XGBoost and exposes real-time predictions through a FastAPI REST API deployed on Render.

## 📌 Project Overview

This project implements a complete end-to-end ML workflow:
🔹 Data preprocessing & feature engineering
🔹 Model training using XGBoost
🔹 Model evaluation & validation
🔹 REST API development using FastAPI
🔹 Cloud deployment for real-time inference

The system helps businesses proactively identify customers likely to churn and take retention actions.

## 🧠 Model Details

Algorithm: XGBoost Classifier
Problem Type: Binary Classification
Target Variable: Churn (0 = Stay, 1 = Leave)
Techniques Used:
Feature Engineering
Train-Test Split
Model Evaluation
Hyperparameter Configuration

### 🏗 System Architecture
User → FastAPI Backend → XGBoost Model → Prediction Response

The trained model is serialized using Pickle and loaded into the FastAPI application for real-time inference.

### 🛠 Tech Stack

#### 💻 Programming
Python
#### 🤖 Machine Learning
XGBoost
Scikit-learn

#### 📊 Data Processing
Pandas
NumPy

#### 🌐 Backend
FastAPI
Uvicorn

#### ☁ Deployment

Render
Git & GitHub

## 📂 Project Structure
Customer-Churn-Prediction/
│
├── api.py
├── train_model.py
├── xgboost_churn_model.pkl
├── churn.csv
├── requirements.txt
└── README.md

### ⚙️ Run Locally
1️⃣ Clone Repository
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Start API Server
python -m uvicorn api:app --reload

Open in browser:

http://127.0.0.1:8000/docs
📡 API Endpoints
🔹 GET /

Health check endpoint

🔹 POST /predict

Predict churn using:
SeniorCitizen
tenure
MonthlyCharges
TotalCharges

## 🌐 Live Deployment

🔗 https://customer-churn-prediction-mt.streamlit.app/

### 📈 Future Improvements

Docker containerization
CI/CD integration
Model monitoring & logging
Advanced feature engineering

### 👩‍💻 Author

Tejaswini Madarapu
GitHub: https://github.com/Tejaswini8888
