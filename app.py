import streamlit as st
import pickle
import numpy as np
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# ---------------- LOAD MODEL & SCALER ----------------
MODEL_PATH = "model (5).pkl"
SCALER_PATH = "minmaxscaler.pkl"   # change to minmaxscaler.pkl if needed

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found in repository")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ standscaler.pkl not found in repository")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# ---------------- CROP LABEL MAPPING ----------------
crop_dict = {
    0: "Rice",
    1: "Maize",
    2: "Chickpea",
    3: "Kidney Beans",
    4: "Pigeon Peas",
    5: "Moth Beans",
    6: "Mung Bean",
    7: "Black Gram",
    8: "Lentil",
    9: "Pomegranate",
    10: "Banana",
    11: "Mango",
    12: "Grapes",
    13: "Watermelon",
    14: "Muskmelon",
    15: "Apple",
    16: "Orange",
    17: "Papaya",
    18: "Coconut",
    19: "Cotton",
    20: "Jute",
    21: "Coffee"
}

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center; color:green;'>🌾 Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Enter soil and climate details to get the best crop</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INPUT FORM ----------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)
    temperature = st.number_input("Temperature (°C)")

with col2:
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

# ---------------- PREDICTION ----------------
st.markdown("")

if st.button("🌱 Recommend Crop"):
    try:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        crop_name = crop_dict[int(prediction[0])]

        st.success(f"✅ Recommended Crop: **{crop_name}**")

        # Confidence score (optional but safe)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)
            confidence = np.max(proba) * 100
            st.info(f"Confidence Level: **{confidence:.2f}%**")

    except Exception as e:
        st.error("⚠️ Error during prediction")
        st.exception(e)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Machine Learning • Smart Agriculture 🌱</p>",
    unsafe_allow_html=True
)
