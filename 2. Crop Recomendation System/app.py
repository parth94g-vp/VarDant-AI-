import streamlit as st
import numpy as np
import joblib

# ===============================
# Load Trained Components
# ===============================

crop_model = joblib.load("crop_recommendation_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

# ===============================
# Page Config
# ===============================

st.set_page_config(page_title="Crop Recommendation", layout="wide")
st.title("🌾 AI-Based Crop Recommendation System")

# ===============================
# Input Section
# ===============================

st.header("🌱 Enter Soil & Environmental Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=500.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=500.0, step=1.0)
    soil_type = st.selectbox("Soil Type", soil_encoder.classes_.tolist())

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)

# ===============================
# Prediction Section
# ===============================

if st.button("🌾 Recommend Crop"):

    try:
        # Encode soil type
        soil_encoded = soil_encoder.transform([soil_type])[0]

        # ⚠ IMPORTANT:
        # Feature order MUST match training order
        input_data = np.array([[ 
            soil_encoded,
            N,
            P,
            K,
            temperature,
            humidity,
            ph
        ]])

        # Make prediction
        prediction = crop_model.predict(input_data)
        crop_name = crop_encoder.inverse_transform(prediction)[0]

        # Save for other modules (if needed)
        st.session_state["recommended_crop"] = crop_name

        # Display result
        st.success(f"✅ Recommended Crop: **{crop_name}**")

        # Optional confidence (if model supports it)
        if hasattr(crop_model, "predict_proba"):
            probabilities = crop_model.predict_proba(input_data)
            confidence = np.max(probabilities) * 100
            st.info(f"📊 Model Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")