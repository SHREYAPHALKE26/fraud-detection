import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define the base directory and model path
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
model_path = os.path.join(base_dir, 'models', 'random_forest_tuned.pkl')

# Load the pre-trained model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit(1)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)  # Reshape for single prediction

        # Make a prediction using the loaded model
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)  # Optional: Get prediction probabilities

        # Return the prediction as a JSON response
        return jsonify({
            'prediction': int(prediction[0]),  # Convert numpy int to Python int
            'prediction_probability': prediction_prob.tolist()  # Optional: Include probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production