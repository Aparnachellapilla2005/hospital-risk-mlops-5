import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    rng = np.random.default_rng(42)
    n = 2000

    age = rng.integers(18, 85, size=n)
    gender = rng.choice(["Male", "Female"], size=n)

    height_cm = np.where(
        gender == "Male",
        rng.normal(172, 7, size=n),
        rng.normal(160, 6, size=n),
    ).round(1)

    weight_kg = np.where(
        gender == "Male",
        rng.normal(78, 12, size=n),
        rng.normal(65, 11, size=n),
    ).round(1)
    weight_kg = np.clip(weight_kg, 40, 150)

    bmi = (weight_kg / ((height_cm / 100) ** 2)).round(1)

    bp = rng.choice(["Low", "Normal", "High"], size=n, p=[0.15, 0.55, 0.30])
    cholesterol = rng.integers(120, 320, size=n)
    glucose = rng.integers(70, 220, size=n)
    smoking = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    exercise = rng.choice(["Low", "Moderate", "High"], size=n, p=[0.35, 0.4, 0.25])

    risk_score = (
        0.03 * (age - 50)
        + 0.08 * (bmi - 25)
        + 0.015 * (cholesterol - 200)
        + 0.02 * (glucose - 110)
        + np.where(bp == "High", 1.2, np.where(bp == "Low", 0.2, 0.0))
        + np.where(smoking == "Yes", 1.0, 0.0)
        + np.where(exercise == "Low", 0.6, np.where(exercise == "High", -0.6, 0.0))
        + rng.normal(0, 1.0, size=n)
    )

    # 1 = HIGH RISK, 0 = LOW RISK
    target = (risk_score > np.median(risk_score)).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "BP": bp,
        "Cholesterol": cholesterol,
        "Glucose": glucose,
        "Smoking": smoking,
        "ExerciseLevel": exercise,
        "BMI": bmi,
        "Target": target,
    })


ENCODERS = {
    "Gender": {"Female": 0, "Male": 1},
    "BP": {"Low": 0, "Normal": 1, "High": 2},
    "Smoking": {"No": 0, "Yes": 1},
    "ExerciseLevel": {"Low": 0, "Moderate": 1, "High": 2},
}


def train_model():
    print("Creating dataset...")
    data = create_dataset()
    data.to_csv("hospital_data.csv", index=False)
    print("Dataset created successfully.")
    print("Number of records:", len(data))

    for column, mapping in ENCODERS.items():
        data[column] = data[column].map(mapping)

    X = data.drop(columns=["Target"])
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "hospital_risk_model.pkl")
    print("\nModel saved as hospital_risk_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }
    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
