import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# ---------------- Page Config ----------------
st.set_page_config(page_title="Weather Prediction App", layout="centered")

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_artifacts():
    model = load_model("weather_model.h5")

    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    return model, scaler

model, scaler = load_artifacts()

# ---------------- UI ----------------
st.title("🌦️ Weather Prediction App")
st.write("Deep Learning based Weather Prediction")

temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

# ---------------- Prediction ----------------
if st.button("Predict Weather"):

    input_data = pd.DataFrame(
        [[temperature, humidity, pressure, wind_speed, rainfall]],
        columns=[
            "Temperature",
            "Humidity",
            "Pressure",
            "Wind Speed",
            "Rainfall"
        ]
    )

    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    prediction = (prediction > 0.5).astype(int)

    if prediction[0][0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.success("☀️ No Rain Expected Tomorrow")

st.markdown("---")
st.caption("Weather Prediction using Deep Learning & Streamlit")
