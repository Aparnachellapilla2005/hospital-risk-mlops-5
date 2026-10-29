
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("hospital_data.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("hospital_risk_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("hospital_risk_model.pkl")

        sample = pd.DataFrame([{
            "Age": 45,
            "Gender": 1,
            "Height_cm": 170.0,
            "Weight_kg": 72.0,
            "BP": 1,
            "Cholesterol": 180,
            "Glucose": 100,
            "Smoking": 0,
            "ExerciseLevel": 1,
            "BMI": 24.9
        }])

        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_risk_patient(self):
        model = joblib.load("hospital_risk_model.pkl")

        sample = pd.DataFrame([{
            "Age": 80,
            "Gender": 1,
            "Height_cm": 170.0,
            "Weight_kg": 110.0,
            "BP": 2,
            "Cholesterol": 300,
            "Glucose": 200,
            "Smoking": 1,
            "ExerciseLevel": 0,
            "BMI": 38.1
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)

    def test_low_risk_patient(self):
        model = joblib.load("hospital_risk_model.pkl")

        sample = pd.DataFrame([{
            "Age": 25,
            "Gender": 0,
            "Height_cm": 160.0,
            "Weight_kg": 50.0,
            "BP": 1,
            "Cholesterol": 130,
            "Glucose": 75,
            "Smoking": 0,
            "ExerciseLevel": 2,
            "BMI": 19.5
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
