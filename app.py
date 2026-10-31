import logging
import sys
import time
import uuid

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("hospital-risk-api")

logger.info(
    "event=application_started service=hospital-risk-prediction"
)

# Track every incoming request
@app.before_request
def start_request_tracking():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


# Log every outgoing response
@app.after_request
def log_request(response):
    duration_ms = (
        time.perf_counter() - g.start_time
    ) * 1000

    response.headers["X-Request-ID"] = g.request_id

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response

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
        logger.error(
            "event=model_file_missing model_file=%s",
            MODEL_PATH
        )

        raise FileNotFoundError(
            "hospital_risk_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH
        )

        return model

    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH
        )
        raise


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

    # Check whether JSON data is provided
    if not data:
        logger.warning(
            "event=prediction_validation_failed request_id=%s reason=missing_json",
            g.request_id
        )
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    # Check for missing fields
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=prediction_validation_failed request_id=%s reason=missing_fields fields=%s",
            g.request_id,
            ",".join(missing_fields)
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    record = {
        feature: data[feature]
        for feature in FEATURES
    }

    # Validate categorical values
    for column, mapping in ENCODERS.items():
        if record[column] not in mapping:
            logger.warning(
                "event=prediction_validation_failed request_id=%s reason=invalid_category field=%s",
                g.request_id,
                column
            )

            return jsonify({
                "error": "Invalid value for " + column,
                "allowed_values": list(mapping.keys())
            }), 400

        record[column] = mapping[record[column]]

    sample = pd.DataFrame([record])

    # Validate numeric values
    try:
        sample = sample.astype(float)

    except (TypeError, ValueError):
        logger.warning(
            "event=prediction_validation_failed request_id=%s reason=invalid_numeric_value",
            g.request_id
        )

        return jsonify({
            "error": "Numeric fields must contain numbers"
        }), 400

    # Perform prediction
    try:
        model = load_model()

        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "HIGH RISK"
            if prediction_code == 1
            else "LOW RISK"
        )

        logger.info(
            "event=prediction_completed request_id=%s prediction=%s prediction_code=%s",
            g.request_id,
            prediction,
            prediction_code
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
