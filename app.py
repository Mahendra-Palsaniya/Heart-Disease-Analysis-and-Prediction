import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("heart.csv")

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
st.sidebar.title("⚙️ Model Selection")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

# -----------------------------
# Train Model
# -----------------------------
if model_option == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# UI
# -----------------------------
st.title("❤️ Heart Disease Prediction App")

st.write("### 🔍 Enter Patient Details")

# INPUTS
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar >120 (1=True, 0=False)", [1, 0])
restecg = st.slider("Rest ECG (0–2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate", value=150)
exang = st.selectbox("Exercise Induced Angina", [1, 0])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.slider("Slope (0–2)", 0, 2, 1)
ca = st.slider("Major Vessels (0–3)", 0, 3, 0)
thal = st.slider("Thal (0–3)", 0, 3, 1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.write(f"💡 Disease Probability: {prob[0][1]:.2f}")

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")
