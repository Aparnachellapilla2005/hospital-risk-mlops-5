from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("hospital_risk_model.pkl")

# Must be in the same order as the columns used for training
FEATURES = [
    "Age",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "BP",
    "Cholesterol",
    "Glucose",
    "Smoking",
    "ExerciseLevel",
    "BMI"
]

# Same text-to-number mapping as train_model.py
ENCODERS = {
    "Gender": {"Female": 0, "Male": 1},
    "BP": {"Low": 0, "Normal": 1, "High": 2},
    "Smoking": {"No": 0, "Yes": 1},
    "ExerciseLevel": {"Low": 0, "Moderate": 1, "High": 2},
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "hospital_risk_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
@app.get("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "hospital-risk-prediction",
        "version": "2.0"
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    record = {feature: data[feature] for feature in FEATURES}

    for column, mapping in ENCODERS.items():
        if record[column] not in mapping:
            return jsonify({
                "error": "Invalid value for " + column,
                "allowed_values": list(mapping.keys())
            }), 400
        record[column] = mapping[record[column]]

    sample = pd.DataFrame([record])

    try:
        sample = sample.astype(float)
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric fields must contain numbers"}), 400

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    prediction = "HIGH RISK" if prediction_code == 1 else "LOW RISK"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
