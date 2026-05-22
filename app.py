import warnings

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import joblib
import numpy as np

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load(
    "loan_approval_pipeline.pkl"
)

# =====================================================
# CREATE FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# HOME ROUTE
# =====================================================

@app.route('/')

def home():

    return """
    <h1>Loan Approval Prediction API Running</h1>
    <p>Use /predict endpoint for predictions</p>
    """

# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.route('/predict', methods=['POST'])

def predict():

    try:

        # Receive JSON Data
        data = request.json

        # Convert Input Into NumPy Array
        features = np.array([[
            data['no_of_dependents'],
            data['education'],
            data['self_employed'],
            data['income_annum'],
            data['loan_amount'],
            data['loan_term'],
            data['cibil_score'],
            data['residential_assets_value'],
            data['commercial_assets_value'],
            data['luxury_assets_value'],
            data['bank_asset_value']
        ]])

        # Prediction
        prediction = model.predict(features)

        # Probability
        probability = model.predict_proba(features)

        # Result
        result = "Loan Approved"

        if prediction[0] == 0:
            result = "Loan Rejected"

        # Confidence
        confidence = round(
            np.max(probability) * 100,
            2
        )

        # Return JSON Response
        return jsonify({
            'prediction': result,
            'confidence': str(confidence) + "%"
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        })

# =====================================================
# RUN FLASK APP
# =====================================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )