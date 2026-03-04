import streamlit as st
import serial
import time
import pandas as pd
from collections import deque

# -------- SETTINGS --------
PORT = "COM4"   # Change to your Arduino COM port
BAUD = 9600
SOIL_THRESHOLD = 600

# -------- SERIAL CONNECTION --------
arduino = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# -------- STREAMLIT CONFIG --------
st.set_page_config(page_title="Smart Irrigation Dashboard", layout="wide")
st.title("🌾 Smart Irrigation System Dashboard")

# -------- PLACEHOLDERS --------
col1, col2, col3, col4 = st.columns(4)

temp_card = col1.empty()
hum_card = col2.empty()
soil_card = col3.empty()
pump_card = col4.empty()

chart_placeholder = st.empty()

# -------- DATA STORAGE --------
temp_data = deque(maxlen=20)
hum_data = deque(maxlen=20)
soil_data = deque(maxlen=20)

while True:
    if arduino.in_waiting > 0:
        line = arduino.readline().decode().strip()

        if line:
            try:
                temperature, humidity, soil = line.split(",")
                temperature = float(temperature)
                humidity = float(humidity)
                soil = int(soil)

                # Decision Logic
                if soil > SOIL_THRESHOLD:
                    arduino.write(b'1')
                    pump_status = "🟢 ON"
                else:
                    arduino.write(b'0')
                    pump_status = "🔴 OFF"

                # Store Data
                temp_data.append(temperature)
                hum_data.append(humidity)
                soil_data.append(soil)

                # -------- UPDATE CARDS --------
                temp_card.metric("🌡 Temperature (°C)", temperature)
                hum_card.metric("💧 Humidity (%)", humidity)
                soil_card.metric("🌱 Soil Moisture", soil)
                pump_card.metric("⚡ Pump Status", pump_status)

                # -------- UPDATE GRAPH --------
                df = pd.DataFrame({
                    "Temperature": temp_data,
                    "Humidity": hum_data,
                    "Soil Moisture": soil_data
                })

                chart_placeholder.line_chart(df)

            except:
                pass

    time.sleep(1)