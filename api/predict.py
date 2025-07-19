from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize the Flask application. Vercel will discover this 'app' object.
app = Flask(__name__)

# --- Model and Scaler Loading ---
# Construct paths to the model files, assuming they are in the root directory
# when deployed on Vercel.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'water_potability_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.joblib')

# Load the pre-trained model and scaler once when the function starts up.
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model/scaler: {e}")
    # Set to None so we can handle the error in the request
    model = None
    scaler = None

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to predict water potability.
    """
    # Check if the model and scaler were loaded correctly
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not found. Check server logs.'}), 500

    try:
        # Get JSON data from the POST request
        data = request.get_json(force=True)
        
        # The order of features must exactly match the order used during training
        # The keys in the JSON from the frontend must match these names.
        feature_keys = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        
        # Create a list of feature values in the correct order
        features = [data[key] for key in feature_keys]
        
        # Convert the features into a pandas DataFrame for the scaler and model
        input_df = pd.DataFrame([features], columns=feature_keys)
        
        # Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # --- Make Prediction ---
        prediction_val = model.predict(input_scaled)[0]
        # Get the probability of the positive class (1 = Potable)
        probability_potable = model.predict_proba(input_scaled)[0][1]

        # --- Determine Potability Status and Level ---
        if prediction_val == 0:
            potability_status = "Not Potable"
            # Not potable water is always considered the lowest level
            level = "Level 1"
        else:
            potability_status = "Potable"
            # Classify into levels based on the model's confidence (probability)
            if probability_potable > 0.85:
                level = "Level 3"  # High confidence
            elif probability_potable > 0.65:
                level = "Level 2"  # Medium confidence
            else:
                level = "Level 1"  # Low confidence, but still predicted as potable
        
        # --- Create Result JSON ---
        result = {
            "Prediction": potability_status,
            "Probability of being Potable": f"{probability_potable:.2%}",
            "Potability Level": level
        }
        
        # Return the successful prediction as a JSON response
        return jsonify(result)

    except KeyError as e:
        # This error occurs if a required key is missing from the input JSON
        return jsonify({'error': f'Missing value for feature: {str(e)}'}), 400
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

# This allows the script to be run locally for testing, though Vercel runs it as a WSGI app.
if __name__ == '__main__':
    # Note: For local testing, ensure your model/scaler files are in the parent directory.
    app.run(debug=True)
