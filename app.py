# app.py - Diabetes Prediction App (Random Forest)
# ------------------------------------------------
# ✅ Password login
# ✅ Safe model loading
# ✅ Predicts diabetes from user input
# ✅ Clean and deploy-ready for Streamlit Cloud

import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. PASSWORD PROTECTION ---
PASSWORD = "diabetes2025"  # 👈 Change this to your own secure password

st.sidebar.header("🔐 Login Required")
password_input = st.sidebar.text_input("Enter Password:", type="password")

if password_input != PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# --- 2. LOAD TRAINED RANDOM FOREST MODEL ---
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "RandomForest_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()

    model = joblib.load(model_path)
    return model


# Load model
model = load_model()

# --- 3. APP TITLE ---
st.title("🩺 Diabetes Prediction App")
st.markdown("This app uses a trained **Random Forest model** to predict the likelihood of diabetes based on patient details.")

# --- 4. INPUT FIELDS ---
st.header("Enter Patient Details:")

Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin Level", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.
