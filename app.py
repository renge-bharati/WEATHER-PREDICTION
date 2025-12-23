import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Weather Prediction App", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Case: model + scaler saved as dictionary
    if isinstance(data, dict):
        model = data.get("model")
        scaler = data.get("scaler")
    else:
        # Case: only model saved
        model = data
        scaler = None

    return model, scaler

model, scaler = load_artifacts()

st.title("🌦️ Weather Prediction App")

st.sidebar.header("Input Weather Parameters")

temperature = st.sidebar.number_input("Temperature (°C)", 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 70.0)
pressure = st.sidebar.number_input("Pressure (hPa)", 1013.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", 10.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0)

if st.button("Predict Weather"):
    input_data = np_
