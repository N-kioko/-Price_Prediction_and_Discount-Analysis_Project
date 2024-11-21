import streamlit as st
import numpy as np
import joblib

# Load the trained model (replace 'random_forest_model.joblib' with your actual file path)
model = joblib.load('random_forest_model.joblib')

# List of all features (must match your model's training data)
feature_names = [
    'Screen Size', 'RAM', 'ROM', 'Warranty', 'Camera', 'Battery Power', 'Number of SIMs',
    'Brand_Infinix Hot', 'Brand_Infinix Hot 40I', 'Brand_Infinix Smart 8', 'Brand_Itel A18',
    'Brand_Itel S23', 'Brand_Oale Pop 8', 'Brand_Oppo A17K', 'Brand_Oppo Aclear83 4Gb Ram',
    'Brand_Samsung Galaxy A05', 'Brand_Samsung Galaxy A05S', 'Brand_Samsung Galaxy A15',
    'Brand_Tecno Pop 8', 'Brand_Tecno Pova 6 Neo', 'Brand_Tecno Spark', 'Brand_Tecno Spark 20',
    'Brand_Tecno Spark 20C', 'Brand_Villaon V20 Se', 'Brand_Xiaomi Redmi 13C',
    'Brand_Xiaomi Redmi 14C', 'Brand_Xiaomi Redmi A3', 'Brand_Xiaomi Redmi Note 13',
    'Color_Black', 'Color_Blue', 'Color_Crystal Green', 'Color_Cyber White',
    'Color_Elemental Blue', 'Color_Energetic Orange', 'Color_Gravity Black',
    'Color_Luxurious Gold', 'Color_Midnight Black', 'Color_Mystery White',
    'Color_Navy Blue', 'Color_Shiny Gold', 'Color_Silver', 'Color_Speed Black',
    'Color_Starry Black', 'Color_Unknown'
]

# Define options for dropdown menus
brands = [
    "Infinix Hot", "Infinix Hot 40I", "Infinix Smart 8", "Itel A18", "Itel S23", "Oale Pop 8",
    "Oppo A17K", "Oppo A83 4Gb Ram", "Samsung Galaxy A05", "Samsung Galaxy A05S",
    "Samsung Galaxy A15", "Tecno Pop 8", "Tecno Pova 6 Neo", "Tecno Spark", "Tecno Spark 20",
    "Tecno Spark 20C", "Villaon V20 Se", "Xiaomi Redmi 13C", "Xiaomi Redmi 14C",
    "Xiaomi Redmi A3", "Xiaomi Redmi Note 13"
]

colors = [
    "Black", "Blue", "Crystal Green", "Cyber White", "Elemental Blue", "Energetic Orange",
    "Gravity Black", "Luxurious Gold", "Midnight Black", "Mystery White", "Navy Blue",
    "Shiny Gold", "Silver", "Speed Black", "Starry Black", "Unknown"
]

# Streamlit UI
st.title("Phone Price Prediction")

# Create columns for side-by-side layout
col1, col2 = st.columns(2)

# Input fields in the first column
with col1:
    brand = st.selectbox("Select Brand", options=brands)
    screen_size = st.number_input("Screen Size (inches)", min_value=0.0, step=0.1)
    ram = st.number_input("RAM (GB)", min_value=0, step=1)
    rom = st.number_input("ROM (GB)", min_value=0, step=1)
    camera = st.number_input("Camera (MP)", min_value=0, step=1)

# Input fields in the second column
with col2:
    color = st.selectbox("Select Color", options=colors)
    warranty = st.number_input("Warranty (Years)", min_value=0, step=1)
    battery_power = st.number_input("Battery Power (mAh)", min_value=0, step=1)
    sim_count = st.number_input("Number of SIMs", min_value=0, step=1)

if st.button("Predict Price"):
    try:
        # One-hot encoding for brand and color
        brand_features = [1 if f"Brand_{brand}" == col else 0 for col in feature_names if "Brand_" in col]
        color_features = [1 if f"Color_{color}" == col else 0 for col in feature_names if "Color_" in col]

        # Create the feature vector
        features = [screen_size, ram, rom, warranty, camera, battery_power, sim_count] + brand_features + color_features

        # Ensure the feature vector matches the model's input
        if len(features) != len(feature_names):
            raise ValueError("Feature vector length mismatch.")

        # Predict
        prediction = model.predict([features])[0]
        st.success(f"Predicted Phone Price: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.button("Refresh"):
    st.rerun()
