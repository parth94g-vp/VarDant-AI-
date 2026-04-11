import streamlit as st
import numpy as np
import joblib
import requests

# ===============================
# Load Models
# ===============================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "crop_recommendation_model.pkl")

crop_model = joblib.load(model_path)
crop_encoder = joblib.load(os.path.join(BASE_DIR, "crop_encoder.pkl"))
soil_encoder = joblib.load(os.path.join(BASE_DIR, "soil_encoder.pkl"))
# ===============================
# Firebase URL
# ===============================
FIREBASE_URL = "https://crop-sensor-default-rtdb.firebaseio.com/sensor.json"

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Crop Recommendation", layout="wide")
st.title("🌾 AI-Based Crop Recommendation System")
st.info("🤖 Automatic Mode Enabled (Live Sensor Data)")

# ===============================
# Session State
# ===============================
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0

if "humidity" not in st.session_state:
    st.session_state.humidity = 0.0

# ===============================
# Fetch Data from Firebase
# ===============================
def fetch_firebase_data():
    try:
        response = requests.get(FIREBASE_URL)
        data = response.json()

        if data:
            return data.get("temperature"), data.get("humidity")
        else:
            return None, None
    except:
        return None, None

# ===============================
# AUTO FETCH (always runs)
# ===============================
temp, hum = fetch_firebase_data()

if temp is not None:
    st.session_state.temperature = temp
    st.session_state.humidity = hum
    st.success(f"📡 Live Data: {temp}°C , {hum}%")
else:
    st.warning("⚠️ Waiting for sensor data...")

# ===============================
# Input Section
# ===============================
st.header("🌱 Enter Soil & Environmental Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0.0, 500.0)
    P = st.number_input("Phosphorus (P)", 0.0, 500.0)
    K = st.number_input("Potassium (K)", 0.0, 500.0)
    soil_type = st.selectbox("Soil Type", soil_encoder.classes_.tolist())

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        value=st.session_state.temperature
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=st.session_state.humidity
    )

    ph = st.number_input("Soil pH", 0.0, 14.0)

# ===============================
# Prediction
# ===============================
if st.button("🌾 Recommend Crop"):

    try:
        soil_encoded = soil_encoder.transform([soil_type])[0]

        input_data = np.array([[ 
            soil_encoded,
            N,
            P,
            K,
            temperature,
            humidity,
            ph
        ]])

        prediction = crop_model.predict(input_data)
        crop_name = crop_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Recommended Crop: **{crop_name}**")

        if hasattr(crop_model, "predict_proba"):
            prob = crop_model.predict_proba(input_data)
            confidence = np.max(prob) * 100
            st.info(f"📊 Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error: {e}")