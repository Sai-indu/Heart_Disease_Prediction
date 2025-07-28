from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("xgb_heart_model.pkl")

# Define the order of feature names expected by the model
FEATURE_NAMES = [
    'age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol',
    'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
    'exercise_angina', 'oldpeak', 'st_slope'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert form inputs to float, in correct order
        input_data = [float(request.form[feature]) for feature in FEATURE_NAMES]

        # Predict using the model
        prediction = model.predict([np.array(input_data)])
        result = "✅ Heart Disease Detected" if prediction[0] == 1 else "🟢 No Heart Disease"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
