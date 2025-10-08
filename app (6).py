
import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
@st.cache_resource
def load_model():
    return joblib.load("RandomForest_model.pkl")

model = load_model()

# Streamlit app
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🌿 Diabetes Prediction App (Random Forest)")

st.write("This app uses a Random Forest model you trained to predict diabetes based on patient details.")

# Input form - Updated to match training data features
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("AGE", min_value=1, max_value=120, value=30)
    gender = st.selectbox("GENDER", ['F', 'M'])
    weight = st.number_input("WEIGHT(kg)", min_value=0.0, max_value=300.0, value=70.0)
with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    blood_sugar = st.number_input("BLOOD SUGAR(mmol/L)", min_value=0.0, max_value=40.0, value=8.0)
    htn = st.selectbox("HTN", [0.0, 1.0])
with col3:
    systolic_bp = st.number_input("SYSTOLIC BP", min_value=0, max_value=300, value=120)
    diastolic_bp = st.number_input("DIASTOLIC BP", min_value=0, max_value=200, value=80)
    treatment = st.selectbox("TREATMENT CATEGORY", ['ab', 'abe', 'ad', 'ae', 'ace', 'other'])


# Make dataframe for prediction
# Create a dictionary with the raw input data
input_dict_raw = {
    'AGE': [age],
    'GENDER': [gender],
    'WEIGHT(kg)': [weight],
    'BMI': [bmi],
    'BP(mmHg)': [f'{systolic_bp}/{diastolic_bp}'], # Reconstruct BP string
    'BLOOD SUGAR(mmol/L)': [blood_sugar],
    'HTN': [htn],
    'TREATMENT_CATEGORY': [treatment] # Use the categorized treatment
}

# Convert to DataFrame
input_df_raw = pd.DataFrame(input_dict_raw)

# Apply the same preprocessing steps as the training data

# Split the 'BP(mmHg)' column
bp_split = input_df_raw['BP(mmHg)'].str.split('/', expand=True)
input_df_raw['SYSTOLIC BP'] = pd.to_numeric(bp_split[0], errors='coerce')
input_df_raw['DIASTOLIC BP'] = pd.to_numeric(bp_split[1], errors='coerce')

# Drop the original 'BP(mmHg)' column
input_df_processed = input_df_raw.drop('BP(mmHg)', axis=1)

# Handle categorical variables using one-hot encoding
input_df_processed = pd.get_dummies(input_df_processed, columns=['GENDER', 'TREATMENT_CATEGORY'], drop_first=True)

# Ensure all columns from training data are present, adding missing ones with default (0)
# and reorder columns to match training data (X_train)
# Assuming X_train.columns is available from previous execution
# If not, you would need to get it from the trained model or save it previously
# This list should match the columns of X_train_imputed used for model training
X_train_cols = ['AGE', 'WEIGHT(kg)', 'BMI', 'BLOOD SUGAR(mmol/L)', 'HTN',
       'SYSTOLIC BP', 'DIASTOLIC BP', 'GENDER_M', 'TREATMENT_CATEGORY_abe',
       'TREATMENT_CATEGORY_ace', 'TREATMENT_CATEGORY_ad',
       'TREATMENT_CATEGORY_ae', 'TREATMENT_CATEGORY_other'] # Updated based on preprocessing in modeling cell


for col in X_train_cols:
    if col not in input_df_processed.columns:
        input_df_processed[col] = 0

input_df_aligned = input_df_processed[X_train_cols]


# Predict button
if st.button("Predict"):
    try:
        pred = model.predict(input_df_aligned)[0]
        prob = model.predict_proba(input_df_aligned)[0][1]

        st.subheader("🔍 Prediction Result")
        # Mapping outcome to label
        outcome_mapping = {
            0: "Non-Diabetic",
            1: "Diabetic"
        }
        st.write("**Outcome:**", outcome_mapping.get(pred, "Unknown"))
        st.write("**Probability of Diabetes:**", f"{prob:.3f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
