import serial
import time

# Change COM port according to your system
# Example: "COM3" for Windows
# Example: "/dev/ttyUSB0" for Linux
arduino = serial.Serial('COM4', 9600, timeout=1)

time.sleep(2)  # Wait for connection to establish

print("Python Brain Started...\n")

SOIL_THRESHOLD = 600

while True:
    if arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()

        if line:
            try:
                temperature, humidity, soil = line.split(",")
                
                temperature = float(temperature)
                humidity = float(humidity)
                soil = int(soil)

                print("---------------------------")
                print(f"Temperature: {temperature} °C")
                print(f"Humidity: {humidity} %")
                print(f"Soil Value: {soil}")

                # Decision Logic
                if soil > SOIL_THRESHOLD:
                    print("Soil Dry → Sending Pump ON")
                    arduino.write(b'1')
                else:
                    print("Soil Wet → Sending Pump OFF")
                    arduino.write(b'0')

            except:
                pass

    time.sleep(1)