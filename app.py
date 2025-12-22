import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Weather Prediction", layout="centered")

@st.cache_resource
def load_files():
    with open("processed_data.pkl", "rb") as f:
        model = pickle.load(f)

    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    return model, scaler

model, scaler = load_files()

st.title("🌦️ Weather Prediction App")

temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[temperature, humidity, pressure, wind_speed, rainfall]],
        columns=["Temperature", "Humidity", "Pressure", "Wind Speed", "Rainfall"]
    )

    # ✅ Apply scaler ONLY if it exists
    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.success("☀️ No Rain Expected Tomorrow")
