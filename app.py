import streamlit as st
import numpy as np
import pickle
import os

# Page config
st.set_page_config(page_title="Weather Prediction App", layout="centered")

# Load model & scaler
@st.cache_resource
def load_artifacts():
    if not os.path.exists("weather_model.pkl"):
        st.error("❌ weather_model.pkl not found")
        st.stop()

    if not os.path.exists("scaler.pkl"):
        st.error("❌ scaler.pkl not found")
        st.stop()

    with open("weather_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()

# App UI
st.title("🌦️ Weather Prediction App")
st.write("Predict tomorrow's weather using Machine Learning")

st.sidebar.header("Input Weather Parameters")

temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", value=70.0)
pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)

if st.button("Predict Weather"):
    input_data = np.array([[temperature, humidity, pressure, wind_speed, rainfall]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.success("☀️ No Rain Expected Tomorrow")


