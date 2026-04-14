import streamlit as st
import requests

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Weather Advisory", layout="wide")

API_KEY = "2d9c1999d5b29778c84d96bbe1056bcd"
CITY = "Mumbai"
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_weather_data():
    params = {
        "q": CITY,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()


def analyze_weather(data):
    rain = False
    heat = False
    pest = "Low"

    for forecast in data["list"][:12]:
        temp = forecast["main"]["temp"]
        humidity = forecast["main"]["humidity"]
        weather = forecast["weather"][0]["main"].lower()

        if "rain" in weather:
            rain = True

        if temp >= 35:
            heat = True

        if humidity >= 80:
            pest = "High"
        elif humidity >= 60:
            pest = "Moderate"

    return rain, heat, pest


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align:center;'>🌾 Smart Weather Advisory</h1>
<p style='text-align:center;color:gray;'>AI-powered farming insights</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# CITY DISPLAY
# -----------------------------
st.markdown(f"""
<div style='text-align:center;font-size:20px;margin-bottom:20px;'>
📍 <b>City:</b> {CITY}
</div>
""", unsafe_allow_html=True)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Get Advisory"):

    data = get_weather_data()

    if data["cod"] != "200":
        st.error("❌ Error fetching weather data")
    else:
        rain, heat, pest = analyze_weather(data)

        col1, col2, col3 = st.columns(3)

        # 🌧 Rain / No Rain Card (UPDATED)
        with col1:
            if rain:
                st.markdown("""
                <div style='background:linear-gradient(135deg,#ff9a9e,#fad0c4);
                padding:25px;border-radius:20px;text-align:center;color:black;
                box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
                <h2>🌧 Rain Alert</h2>
                <p>Rain expected</p>
                <b>👉 Delay irrigation</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:linear-gradient(135deg,#56ab2f,#a8e063);
                padding:25px;border-radius:20px;text-align:center;color:white;
                box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
                <h2>🌤 No Rain</h2>
                <p>Safe for irrigation</p>
                </div>
                """, unsafe_allow_html=True)

        # 🔥 Heat Card (same as before)
        with col2:
            if heat:
                st.markdown("""
                <div style='background:linear-gradient(135deg,#ff758c,#ff7eb3);
                padding:25px;border-radius:20px;text-align:center;color:white;
                box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
                <h2>🔥 Heat Stress</h2>
                <p>High temperature detected</p>
                <b>👉 Increase irrigation</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:linear-gradient(135deg,#89f7fe,#66a6ff);
                padding:25px;border-radius:20px;text-align:center;color:white;
                box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
                <h2>🌤 Normal Temp</h2>
                </div>
                """, unsafe_allow_html=True)

        # 🐛 Pest Card (UPDATED COLORS)
        with col3:

            if pest == "Low":
                gradient = "linear-gradient(135deg,#11998e,#38ef7d)"
            elif pest == "Moderate":
                gradient = "linear-gradient(135deg,#f7971e,#ffd200)"
            else:
                gradient = "linear-gradient(135deg,#ff416c,#ff4b2b)"

            st.markdown(f"""
            <div style='background:{gradient};
            padding:25px;border-radius:20px;text-align:center;color:white;
            box-shadow:0px 4px 10px rgba(0,0,0,0.2);'>
            <h2>🐛 Pest Risk</h2>
            <h3>{pest}</h3>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # SUMMARY SECTION
        # -----------------------------
        st.markdown("---")
        st.markdown("### 📊 Summary Insights")

        if rain:
            st.info("🌧 Rain expected → Avoid irrigation")
        if heat:
            st.warning("🔥 Heat stress → Protect crops")
        if pest == "High":
            st.error("🐛 High pest risk → Take action")
        elif pest == "Moderate":
            st.warning("🐛 Moderate pest risk → Monitor crops")
        else:
            st.success("✅ Low pest risk")