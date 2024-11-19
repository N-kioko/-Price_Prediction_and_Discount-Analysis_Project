import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load the trained model
model_filename = 'random_forest_model.pkl'
model = pickle.load(open(model_filename, 'rb'))

# List of all features (should match your model's training data order)
feature_names = [
    'Screen Size', 'RAM', 'ROM', 'Warranty', 'Camera', 'Battery Power', 'Number of SIMs',
    'Brand_Infinix Hot', 'Brand_Infinix Hot 40I', 'Brand_Infinix Smart 8', 'Brand_Itel A18',
    'Brand_Itel S23', 'Brand_Oale Pop 8', 'Brand_Oppo A17K', 'Brand_Oppo A83 4Gb Ram',
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

# Create Flask app
app = Flask(__name__)

# Render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()
        print("Received data:", data)
        
        # Extract fields from the request
        screen_size = float(data['screen_size'])  # Use screen_size instead of 'Screen Size'
        ram = int(data['ram'])
        rom = int(data['rom'])
        warranty = int(data['warranty'])
        camera = int(data['camera'])
        battery = int(data['battery_power'])  # Use battery_power instead of 'Battery Power'
        sims = int(data['sim_count'])  # Use sim_count instead of 'Number of SIMs'

        # Handle brand and color one-hot encoding
        brand = data['brand']
        color = data['color']
        brand_features = [1 if f"Brand_{brand}" == col else 0 for col in feature_names if "Brand_" in col]
        color_features = [1 if f"Color_{color}" == col else 0 for col in feature_names if "Color_" in col]

        # Create the complete feature vector
        features = [screen_size, ram, rom, warranty, camera, battery, sims] + brand_features + color_features

        # Ensure the feature vector matches the model's expected input
        if len(features) != len(feature_names):
            raise ValueError("Feature vector length mismatch.")

        # Predict the price
        predicted_price = model.predict([features])[0]

        # Return the prediction
        print(predicted_price)
        return jsonify({'price': predicted_price})

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=False)

