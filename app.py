import streamlit as st
import pickle
import numpy as np
import os
import urllib.parse

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# -------------------- SAFE FILE CHECK --------------------
if not os.path.exists("model (5).pkl"):
    st.error("❌ model.pkl not found. Please upload it to the GitHub repository.")
    st.stop()

if not os.path.exists("minmaxscaler.pkl"):
    st.error("❌ minmaxscaler.pkl not found. Please upload it to the GitHub repository.")
    st.stop()

model = pickle.load(open("model (5).pkl", "rb"))
scaler = pickle.load(open("minmaxscaler.pkl", "rb"))

# -------------------- CROP LABEL MAPPING --------------------
crop_dict = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidney Beans",
    4: "Pigeon Peas", 5: "Moth Beans", 6: "Mung Bean",
    7: "Black Gram", 8: "Lentil", 9: "Pomegranate",
    10: "Banana", 11: "Mango", 12: "Grapes",
    13: "Watermelon", 14: "Muskmelon", 15: "Apple",
    16: "Orange", 17: "Papaya", 18: "Coconut",
    19: "Cotton", 20: "Jute", 21: "Coffee"
}

# -------------------- REAL IMAGE LINKS (OPTIONAL) --------------------
crop_images = {
    "Rice": "https://images.unsplash.com/photo-1600850056064-a8b380df8395",
    "Maize": "https://images.unsplash.com/photo-1598514982205-faa48c1a2c59",
    "Chickpea": "https://images.unsplash.com/photo-1628695191732-ef20e25c1f8a",
    "Banana": "https://images.unsplash.com/photo-1574226516831-e1dff420e37d",
    "Mango": "https://images.unsplash.com/photo-1580910051074-7e6cda05a36b",
    "Apple": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
    "Orange": "https://images.unsplash.com/photo-1547514701-42782101795e",
    "Grapes": "https://images.unsplash.com/photo-1506806732259-39c2d0268443",
    "Watermelon": "https://images.unsplash.com/photo-1622205313162-be1d5712a43e",
    "Coffee": "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
}

# -------------------- IMAGE FALLBACK FUNCTION (KEY PART) --------------------
def get_crop_image(crop_name):
    """
    Always returns a valid image.
    Uses real image if available, otherwise generates a placeholder image.
    """
    if crop_name in crop_images:
        return crop_images[crop_name]
    else:
        text = urllib.parse.quote(crop_name)
        return f"https://via.placeholder.com/400x300.png?text={text}"

# -------------------- HEADER --------------------
st.markdown(
    "<h1 style='text-align:center; color:green;'>🌾 Smart Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Enter soil and climate details to get the best crop recommendation</p>",
    unsafe_allow_html=True
)

# -------------------- INPUT FORM --------------------
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

# -------------------- PREDICTION --------------------
if st.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop_name = crop_dict[prediction[0]]

    # Result Card
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #56ab2f, #a8e063);
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        ">
            <h2 style="color:white;">🌾 Recommended Crop</h2>
            <h1 style="color:white;">{crop_name}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Crop Image (GUARANTEED)
    st.image(
        get_crop_image(crop_name),
        caption=f"{crop_name} Crop",
        use_column_width=True
    )

# -------------------- DECORATIVE CROP GALLERY --------------------
st.markdown("---")
st.markdown("## 🌾 Decorative Crop Gallery")

cols = st.columns(4)
all_crops = list(crop_dict.values())

for i, crop in enumerate(all_crops):
    with cols[i % 4]:
        st.image(
            get_crop_image(crop),
            caption=crop,
            use_column_width=True
        )

# -------------------- FOOTER --------------------
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Machine Learning • Smart Agriculture • Sustainable Farming 🌱"
    "</p>",
    unsafe_allow_html=True
)
