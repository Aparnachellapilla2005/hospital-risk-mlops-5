
import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("hospital-risk-api")

# Prometheus Metrics
HTTP_REQUESTS = Counter(
    "hospital_api_http_requests_total",
    "Total HTTP requests received by the Hospital Risk Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "hospital_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "hospital_api_predictions_total",
    "Total Hospital Risk Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "hospital_api_model_loaded",
    "Whether the Hospital Risk Prediction model is loaded"
)

logger.info(
    "event=application_started service=hospital-risk-prediction"
)

# Request Tracking
@app.before_request
def start_request_tracking():

    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


# Request Logging and Metrics
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

    # Record Prometheus HTTP metrics
    if request.path != "/metrics":

        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_ms / 1000)

    return response


# Model Configuration
MODEL_PATH = Path("hospital_risk_model.pkl")

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

ENCODERS = {
    "Gender": {"Female": 0, "Male": 1},
    "BP": {"Low": 0, "Normal": 1, "High": 2},
    "Smoking": {"No": 0, "Yes": 1},
    "ExerciseLevel": {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    },
}


# Model Loading
def load_model():

    if not MODEL_PATH.exists():

        MODEL_LOADED.set(0)

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

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH
        )

        return model

    except Exception:

        MODEL_LOADED.set(0)

        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH
        )

        raise


# Health Check Endpoint
@app.get("/")
@app.get("/health")
def health_check():

    return jsonify({
        "status": "ok",
        "service": "hospital-risk-prediction",
        "version": "2.0"
    })


# Prediction Endpoint
@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    # Validate JSON request
    if not data:

        logger.warning(
            "event=prediction_validation_failed request_id=%s reason=missing_json",
            g.request_id
        )

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    # Validate required fields
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

    # Perform model prediction
    try:

        model = load_model()

        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "HIGH RISK"
            if prediction_code == 1
            else "LOW RISK"
        )

        PREDICTIONS.labels(result="success").inc()

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

        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# Prometheus Metrics Endpoint
@app.get("/metrics")
def metrics():

    return Response(
        response=generate_latest(),
        status=200,
        mimetype=CONTENT_TYPE_LATEST
    )


# Run Application
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
