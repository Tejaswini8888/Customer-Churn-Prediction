# 📊 Customer Churn Prediction API  
## 🚀 XGBoost + FastAPI | Production-Ready ML System  

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Render](https://img.shields.io/badge/Render-Deployed-blue)

---

## 📌 Project Overview  

This project implements a complete end-to-end Machine Learning workflow:

- 🔹 Data preprocessing & feature engineering  
- 🔹 Model training using XGBoost  
- 🔹 Model evaluation & validation  
- 🔹 REST API development using FastAPI  
- 🔹 Cloud deployment for real-time inference  

The system helps businesses proactively identify customers likely to churn and take retention actions.

---

## 🧠 Model Details  

- **Algorithm:** XGBoost Classifier  
- **Problem Type:** Binary Classification  
- **Target Variable:**  
  - `0` → Customer Stays  
  - `1` → Customer Leaves  

### Techniques Used  

- Feature Engineering  
- Train-Test Split  
- Model Evaluation  
- Hyperparameter Configuration  

---

## 🏗 System Architecture  

```
User → FastAPI Backend → XGBoost Model → Prediction Response
```

The trained model is serialized using **Pickle** and loaded into the FastAPI application for real-time inference.

---

## 🛠 Tech Stack  

### 💻 Programming  
- Python  

### 🤖 Machine Learning  
- XGBoost  
- Scikit-learn  

### 📊 Data Processing  
- Pandas  
- NumPy  

### 🌐 Backend  
- FastAPI  
- Uvicorn  

### ☁ Deployment  
- Render  
- Git & GitHub  

---

## 📂 Project Structure  

```
Customer-Churn-Prediction/
│
├── api.py
├── train_model.py
├── xgboost_churn_model.pkl
├── churn.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally  

### 1️⃣ Clone Repository  

```bash
git clone https://github.com/Tejaswini8888/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2️⃣ Install Dependencies  

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model  

```bash
python train_model.py
```

### 4️⃣ Start API Server  

```bash
uvicorn api:app --reload
```

Open Swagger UI in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints  

### 🔹 GET `/`  
Health check endpoint  

### 🔹 POST `/predict`  

Predict churn using the following input features:

- `SeniorCitizen`  
- `tenure`  
- `MonthlyCharges`  
- `TotalCharges`  

---

### 📥 Example Request  

```json
{
  "SeniorCitizen": 0,
  "tenure": 24,
  "MonthlyCharges": 70.5,
  "TotalCharges": 1680.2
}
```

---

### 📤 Example Response  

```json
{
  "prediction": 1,
  "churn_probability": 0.7421
}
```

---

## 🌐 Live Deployment  

🔗 https://customer-churn-prediction-mt.streamlit.app/
---

## 📈 Future Improvements  

- Docker containerization  
- CI/CD integration  
- Model monitoring & logging  
- SHAP explainability integration  
- Advanced feature engineering  

---

## 👩‍💻 Author  

**Tejaswini Madarapu**  

🔗 GitHub: https://github.com/Tejaswini8888  
🔗 LinkedIn: https://www.linkedin.com/in/tejaswini-madarapu/  

---

⭐ If you found this project useful, consider giving it a star!
