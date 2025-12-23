import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Weather Prediction App", layout="centered")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Case 1: model + scaler saved as tuple
    if isinstance(data, tuple):
        model, scaler = data

    # Case 2: model + scaler saved as dict
    elif isinstance(data, dict):
        model = data.get("model")
        scaler = data.get("scaler")

    # Case 3: only model saved
    else:
        model = data
        scaler = None

    return model, scaler


model, scaler = load_artifacts()

# ---------- UI ----------
st.title("🌦️ Weather Prediction App")
st.write("Predict tomorrow's weather using Machine Learning")

st.sidebar.header("Input Weather Parameters")

temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", value=70.0)
pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)

# ---------- Prediction ----------
if st.button("Predict Weather"):
    input_data = np.array([[temperature, humidity, pressure, wind_speed, rainfall]])

    # Apply scaler ONLY if it exists
    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.success("☀️ No Rain Expected Tomorrow")



