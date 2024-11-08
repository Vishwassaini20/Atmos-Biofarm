from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your model (ensure the path is correct)
model = joblib.load('your_model.pkl')  # Path to the trained model

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Extract features from the JSON and prepare for prediction
    features = np.array([data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']])
    features = features.reshape(1, -1)  # Reshape for single prediction
    
    # Predict crop recommendation
    prediction = model.predict(features)
    
    # Send prediction back as JSON
    result = {'recommended_crop': prediction[0]}
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
