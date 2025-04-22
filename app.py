# type: ignore
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model and scaler (update these paths if necessary)
model = load_model('C:/Users/ADMIN/Desktop/Projects/ANN_Project/model/final_sonar_model.keras')
scaler = joblib.load('scaler.pkl')

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
    app.run(debug=True)
