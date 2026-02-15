import numpy as np
import joblib

# Load model & encoders
model = joblib.load("fertilizer_category_model.pkl")
soil_encoder = joblib.load("fertilizer_soil_encoder.pkl")
crop_encoder = joblib.load("fertilizer_crop_encoder.pkl")
category_encoder = joblib.load("fertilizer_category_encoder.pkl")

print("\n🧪 Fertilizer Category Prediction")
print("--------------------------------")

soil = input("Enter Soil Type: ")
crop = input("Enter Crop Type: ")

N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorous (P): "))
K = float(input("Enter Potassium (K): "))
temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
moisture = float(input("Enter Moisture: "))

try:
    soil_e = soil_encoder.transform([soil])[0]
    crop_e = crop_encoder.transform([crop])[0]
except ValueError:
    print("❌ Crop or Soil type not supported")
    exit()

input_data = np.array([[soil_e, crop_e, N, P, K, temp, humidity, moisture]])

pred = model.predict(input_data)
category = category_encoder.inverse_transform(pred)

print("\n✅ Recommended Fertilizer Category:", category[0])
print("--------------------------------")