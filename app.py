import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("model (5).pkl", "rb"))
scaler = pickle.load(open("minmaxscaler.pkl", "rb"))  # change if needed

# Crop label mapping
crop_dict = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidney Beans",
    4: "Pigeon Peas", 5: "Moth Beans", 6: "Mung Bean",
    7: "Black Gram", 8: "Lentil", 9: "Pomegranate",
    10: "Banana", 11: "Mango", 12: "Grapes",
    13: "Watermelon", 14: "Muskmelon", 15: "Apple",
    16: "Orange", 17: "Papaya", 18: "Coconut",
    19: "Cotton", 20: "Jute", 21: "Coffee"
}

# Crop images (royalty-free Unsplash)
crop_images = {
    "Rice": "https://images.unsplash.com/photo-1600850056064-a8b380df8395",
    "Maize": "https://images.unsplash.com/photo-1598514982205-faa48c1a2c59",
    "Chickpea": "https://images.unsplash.com/photo-1628695191732-ef20e25c1f8a",
    "Mango": "https://images.unsplash.com/photo-1580910051074-7e6cda05a36b",
    "Banana": "https://images.unsplash.com/photo-1574226516831-e1dff420e37d",
    "Apple": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
    "Orange": "https://images.unsplash.com/photo-1547514701-42782101795e",
    "Grapes": "https://images.unsplash.com/photo-1506806732259-39c2d0268443",
    "Watermelon": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
    "Coffee": "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
}

# Page config
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

st.markdown(
    "<h1 style='text-align:center; color:green;'>🌾 Smart Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

st.write("### 🌱 Enter Soil & Climate Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen", min_value=0)
    P = st.number_input("Phosphorus", min_value=0)
    K = st.number_input("Potassium", min_value=0)
    temperature = st.number_input("Temperature (°C)")

with col2:
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

if st.button("🌾 Recommend Best Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop_name = crop_dict[prediction[0]]

    st.success(f"✅ Recommended Crop: **{crop_name}**")

    if crop_name in crop_images:
        st.image(crop_images[crop_name], caption=crop_name, use_column_width=True)

    st.balloons()
