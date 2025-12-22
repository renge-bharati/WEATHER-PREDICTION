import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Weather Prediction App", layout="centered")

# Load model and scaler
@st.cache_resource
def load_model():
    with open("processed_data.pkl", "rb") as f:
        model = pickle.load(f)
    with open("processed_data.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.title("🌦️ Weather Prediction App")
st.write("Predict tomorrow's weather using machine learning")

st.sidebar.header("Input Weather Parameters")

# Input fields (adjust names if your dataset columns differ)
temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", value=70.0)
pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)

# Predict button
if st.button("Predict Weather"):
    input_data = np.array([[temperature, humidity, pressure, wind_speed, rainfall]])
    if prediction[0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.success("☀️ No Rain Expected Tomorrow")
