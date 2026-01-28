import streamlit as st
import joblib
import numpy as np

# 1. Model aur Scaler load karein 
# (Make sure ye dono files aapke 'Diabetes Prediction Web App' folder ke andar hain)
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Page Configuration
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# 3. Web App UI
st.title("Diabetes Prediction System 🏥")
st.markdown("Enter the patient's clinical data below:")

# 4. Input Fields (2 Columns for better look)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    thickness = st.number_input("Thickness", min_value=0)
with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI (e.g., 22.5)", min_value=0.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)

# 5. Prediction Logic
if st.button("Check Result"):
    # 5. Prediction Logic with Validation
    zero_forbidden = {
        "Glucose": glucose,
        "Blood Pressure": bp,
        "Thickness": thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age
    }
    
    # Check for zero values
    invalid_fields = [name for name, value in zero_forbidden.items() if value == 0]
    
    if invalid_fields:
        st.warning(f"⚠️ Value 0 is not possible for: {', '.join(invalid_fields)}. Please enter a valid non-zero value.")
    else:
        try:
            # Prediction process starts only if all inputs are non-zero
            input_data = [pregnancies, glucose, bp, thickness, insulin, bmi, dpf, age]
            features = np.array([input_data])
            
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            
            if prediction[0] == 1:
                st.error("### Result: The person is likely DIABETIC.")
            else:
                st.success("### Result: The person is NOT diabetic.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    