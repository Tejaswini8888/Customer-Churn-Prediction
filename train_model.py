import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("churn.csv")

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Final check
print("\nFinal dataset shape:", df.shape)
print(df.head())

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)


import pickle

# Save Random Forest model
pickle.dump(rf_model, open("churn_model.pkl", "wb"))

print("Model saved successfully")


from xgboost import XGBClassifier

# Create model
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost Accuracy:", xgb_accuracy)


import pickle

pickle.dump(xgb_model, open("xgboost_churn_model.pkl", "wb"))

print("XGBoost model saved")



import matplotlib.pyplot as plt

importance = xgb_model.feature_importances_
features = X.columns

# Create dataframe
import pandas as pd

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values("Importance", ascending=False).head(10)

# Plot
plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()