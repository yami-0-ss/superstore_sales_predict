import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model safely
MODEL_PATH = "grad_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

# Exact feature names extracted from grad_model.pkl
FEATURE_NAMES = [
    'Ship Mode', 'Customer Name', 'Segment', 'Country', 'City', 
    'State', 'Region', 'Category', 'Sub-Category', 'Product Name', 
    'Sales', 'Quantity', 'Discount'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #a855f7;
            --accent-pink: #ec4899;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            color: var(--text-main);
            overflow-x: hidden;
        }

        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            z-index: 0;
            animation: float 10s infinite ease-in-out alternate;
        }
        .orb-1 { width: 300px; height: 300px; background: #6366f1; top: 10%; left: 15%; }
        .orb-2 { width: 350px; height: 350px; background: #d946ef; bottom: 10%; right: 15%; animation-delay: -5s; }

        @keyframes float {
            0% { transform: translateY(0px) scale(1); }
            100% { transform: translateY(30px) scale(1.08); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 850px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(236, 72, 153, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(236, 72, 153, 0.45);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(168, 85, 247, 0.1);
            border: 1px solid rgba(168, 85, 247, 0.3);
            border-radius: 16px;
            text-align: center;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .error-card {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 16px;
            color: #f87171;
            text-align: center;
        }

        @keyframes popIn {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-card h3 {
            font-size: 1rem;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
        }

        .result-card .prediction-val {
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <header>
            <h1>Predictive Analytics Dashboard</h1>
            <p>Gradient Boosting Regressor Deployment</p>
        </header>

        <form method="POST" action="/predict" class="grid-form">
            {% for feature in features %}
            <div class="input-group">
                <label for="{{ feature }}">{{ feature }}</label>
                <input 
                    type="{% if feature in ['Sales', 'Quantity', 'Discount'] %}number{% else %}text{% endif %}" 
                    step="any" 
                    name="{{ feature }}" 
                    id="{{ feature }}" 
                    placeholder="Enter {{ feature }}" 
                    required>
            </div>
            {% endfor %}

            <button type="submit" class="btn-submit">Calculate Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Predicted Target Value</h3>
            <div class="prediction-val">{{ prediction }}</div>
        </div>
        {% endif %}

        {% if error is not none %}
        <div class="error-card">
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None, error="Model file ('grad_model.pkl') not found or invalid.")

    try:
        data = {}
        for feature in FEATURE_NAMES:
            val = request.form.get(feature, "")
            
            # Numeric conversion for continuous features
            if feature in ['Sales', 'Quantity', 'Discount']:
                data[feature] = [float(val) if val else 0.0]
            else:
                # Deterministic float hash encoding for text inputs
                data[feature] = [float(abs(hash(str(val))) % 10000)]

        # Ensure correct column ordering matching training data
        input_df = pd.DataFrame(data)[FEATURE_NAMES]
        
        # Predict using GradientBoostingRegressor
        prediction_val = model.predict(input_df)[0]
        formatted_pred = f"{float(prediction_val):,.2f}"

        return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=formatted_pred, error=None)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None, error=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
