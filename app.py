import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Weather Prediction App", layout="centered")

@st.cache_resource
def load_model():
    with open("processed_data.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🌦️ Weather Prediction App")

temperature = st.number_input("Temperature (°C)", 25.0)
humidity = st.number_input("Humidity (%)", 70.0)
pressure = st.number_input("Pressure (hPa)", 1013.0)
wind_speed = st.number_input("Wind Speed (km/h)", 10.0)
rainfall = st.number_input("Rainfall (mm)", 0.0)

if st.button("Predict Weather"):
    input_data = np.array([[temperature, humidity, pressure, wind_speed, rainfall]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.success("☀️ No Rain Expected Tomorrow")


