import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Load Saved Models & Encoders
# ----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("fertilizer_category_model.pkl")
    soil_encoder = joblib.load("fertilizer_soil_encoder.pkl")
    crop_encoder = joblib.load("fertilizer_crop_encoder.pkl")
    category_encoder = joblib.load("fertilizer_category_encoder.pkl")
    return model, soil_encoder, crop_encoder, category_encoder

model, soil_encoder, crop_encoder, category_encoder = load_models()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Fertilizer Category Recommendation")
st.write("Enter soil, crop and nutrient details to get recommended fertilizer category.")

st.markdown("---")

# ----------------------------
# User Inputs
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    soil = st.selectbox(
        "Select Soil Type",
        options=soil_encoder.classes_
    )
    
    N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
    P = st.number_input("Phosphorous (P)", min_value=0.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)

with col2:
    crop = st.selectbox(
        "Select Crop Type",
        options=crop_encoder.classes_
    )
    
    temp = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", step=0.1)
    moisture = st.number_input("Moisture (%)", step=0.1)

st.markdown("---")

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔍 Predict Fertilizer Category"):
    try:
        soil_e = soil_encoder.transform([soil])[0]
        crop_e = crop_encoder.transform([crop])[0]

        input_data = np.array([[soil_e, crop_e, N, P, K, temp, humidity, moisture]])

        pred = model.predict(input_data)
        category = category_encoder.inverse_transform(pred)[0]

        st.success(f"✅ Recommended Fertilizer Category: **{category}**")

    except Exception as e:
        st.error("❌ Error in prediction. Please check inputs.")
        st.exception(e)
