import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model (5).pkl", "rb"))
scaler = pickle.load(open("standscaler.pkl", "rb"))  # or minmaxscaler.pkl

st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌱 Crop Recommendation System")
st.write("Enter soil and climate details to get the best crop recommendation")

# Input fields
N = st.number_input("Nitrogen", min_value=0)
P = st.number_input("Phosphorus", min_value=0)
K = st.number_input("Potassium", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Recommend Crop 🌾"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"✅ Recommended Crop: **{prediction[0]}**")
