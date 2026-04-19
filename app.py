import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engineer import FeatureEngineer

# ===== Load Pipeline =====
pipeline = joblib.load("heart_pipeline.pkl")

st.title("🫀 Heart Disease Prediction System")

# ===== Patient Demographics =====
st.subheader("Patient Demographics")
age  = st.number_input("Age (years)", 1, 120, 52)
sex  = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
fbs  = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
exang = st.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", [0, 1])

# ===== Clinical Measurements =====
st.subheader("Clinical Measurements")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 125)
chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 212)
thalach  = st.number_input("Max Heart Rate Achieved (bpm)", 60, 250, 168)
oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)

# ===== Diagnostic Results =====
st.subheader("Diagnostic Results")
cp = st.selectbox("Chest Pain Type",
                  [0, 1, 2, 3],
                  format_func=lambda x: {0:"0 – Typical Angina",
                                         1:"1 – Atypical Angina",
                                         2:"2 – Non-Anginal Pain",
                                         3:"3 – Asymptomatic"}[x])

restecg = st.selectbox("Resting ECG Results",
                       [0, 1, 2],
                       format_func=lambda x: {0:"0 – Normal",
                                              1:"1 – ST-T Wave Abnormality",
                                              2:"2 – Left Ventricular Hypertrophy"}[x])

slope = st.selectbox("Slope of Peak Exercise ST Segment",
                     [0, 1, 2],
                     format_func=lambda x: {0:"0 – Upsloping",
                                            1:"1 – Flat",
                                            2:"2 – Downsloping"}[x])

ca   = st.selectbox("Major Vessels Coloured by Fluoroscopy (0–4)", [0, 1, 2, 3, 4])

thal = st.selectbox("Thalassemia",
                    [0, 1, 2, 3],
                    format_func=lambda x: {0:"0 – Normal",
                                           1:"1 – Fixed Defect",
                                           2:"2 – Reversible Defect",
                                           3:"3 – Unknown"}[x])

# ===== Prediction =====
if st.button("Predict Heart Disease"):
    input_df = pd.DataFrame([{
        "age":      age,
        "sex":      sex,
        "cp":       cp,
        "trestbps": trestbps,
        "chol":     chol,
        "fbs":      fbs,
        "restecg":  restecg,
        "thalach":  thalach,
        "exang":    exang,
        "oldpeak":  oldpeak,
        "slope":    slope,
        "ca":       ca,
        "thal":     thal,
    }])

    pred  = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    st.write("🌲 Random Forest:",
             "⚠️ Heart Disease Detected" if pred == 1 else "✅ No Heart Disease Detected")
    st.write(f"Disease Probability: {proba[1]*100:.1f}%  |  No Disease Probability: {proba[0]*100:.1f}%")
