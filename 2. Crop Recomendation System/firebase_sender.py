# This script will:
# ✔ Read Arduino data
# ✔ Send it to Firebase every 2 sec

import serial
import time
import firebase_admin
from firebase_admin import credentials, db
import math

# ===============================
# Firebase Setup
# ===============================
cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://crop-sensor-default-rtdb.firebaseio.com/'
})

ref = db.reference("sensor")

# ===============================
# Read Sensor
# ===============================
def read_sensor():
    try:
        ser = serial.Serial('COM4', 9600, timeout=2)
        time.sleep(2)

        line = ser.readline().decode().strip()
        ser.close()

        if line:
            parts = line.split(",")

            if len(parts) == 3:
                temp = float(parts[0])
                hum = float(parts[1])

                # ignore NaN
                if math.isnan(temp) or math.isnan(hum):
                    return None, None

                return temp, hum

        return None, None

    except:
        return None, None


# ===============================
# Loop
# ===============================
while True:
    temp, hum = read_sensor()

    if temp is not None:
        data = {
            "temperature": temp,
            "humidity": hum
        }

        ref.set(data)
        print("Sent:", data)

    time.sleep(2)