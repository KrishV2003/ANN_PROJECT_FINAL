# type: ignore
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Create Flask app
app = Flask(__name__)

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to your model & scaler inside the repo
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'final_sonar_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Load once at startup
model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        df = pd.read_csv(file, header=None)
        
        # Check if the DataFrame is (60, 1) (i.e., 60 values, one per row) then transpose it
        if df.shape[0] == 60 and df.shape[1] == 1:
            df = df.transpose()  # Now shape is (1, 60)

        X = df.values.astype(np.float32)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)
        # Use the first prediction for simplicity
        prob = float(pred[0][0])
        label = 'Mine' if prob > 0.5 else 'Rock'

        return jsonify({
            'label': label,
            'confidence': {'rock': 1 - prob, 'mine': prob}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Only for local testing—Render uses Gunicorn via Procfile
    app.run(debug=True)
