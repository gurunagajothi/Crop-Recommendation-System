import streamlit as st
import pickle
import numpy as np

# -------------------- Load Model & Scaler --------------------
model = pickle.load(open("model (5).pkl", "rb"))
scaler = pickle.load(open("minmaxscaler.pkl", "rb"))  # change if needed

# -------------------- Crop Label Mapping --------------------
crop_dict = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidney Beans",
    4: "Pigeon Peas", 5: "Moth Beans", 6: "Mung Bean",
    7: "Black Gram", 8: "Lentil", 9: "Pomegranate",
    10: "Banana", 11: "Mango", 12: "Grapes",
    13: "Watermelon", 14: "Muskmelon", 15: "Apple",
    16: "Orange", 17: "Papaya", 18: "Coconut",
    19: "Cotton", 20: "Jute", 21: "Coffee"
}

# -------------------- Crop Images (Decorative Gallery) --------------------
crop_images = {
    "Rice": "https://images.unsplash.com/photo-1600850056064-a8b380df8395",
    "Maize": "https://images.unsplash.com/photo-1598514982205-faa48c1a2c59",
    "Chickpea": "https://images.unsplash.com/photo-1628695191732-ef20e25c1f8a",
    "Banana": "https://images.unsplash.com/photo-1574226516831-e1dff420e37d",
    "Mango": "https://images.unsplash.com/photo-1580910051074-7e6cda05a36b",
    "Apple": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
    "Grapes": "https://images.unsplash.com/photo-1506806732259-39c2d0268443",
    "Orange": "https://images.unsplash.com/photo-1547514701-42782101795e",
    "Watermelon": "https://images.unsplash.com/photo-1622205313162-be1d5712a43e",
    "Coffee": "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
}

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center; color:green;'>🌾 Smart Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

st.write("### 🌱 Enter Soil & Climate Details")

# -------------------- Input Section --------------------
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

# -------------------- Prediction --------------------
if st.button("🌾 Recommend Best Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop_name = crop_dict[prediction[0]]

    # Result Card
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #a8e063, #56ab2f);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
        ">
            <h2 style="color:white;">🌱 Recommended Crop</h2>
            <h1 style="color:white;">{crop_name}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Crop Image
    if crop_name in crop_images:
        st.image(
            crop_images[crop_name],
            caption=f"{crop_name} Crop",
            use_column_width=True
        )

# -------------------- Decorative Crop Gallery --------------------
st.markdown("---")
st.markdown("## 🌾 Decorative Crop Gallery")

gallery_crops = list(crop_images.keys())

cols = st.columns(4)
for idx, crop in enumerate(gallery_crops):
    with cols[idx % 4]:
        st.image(
            crop_images[crop],
            caption=crop,
            use_column_width=True
        )

st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Smart Agriculture • Machine Learning • Sustainable Farming 🌱"
    "</p>",
    unsafe_allow_html=True
)
