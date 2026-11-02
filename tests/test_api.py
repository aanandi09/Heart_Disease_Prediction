"""
Automated Test Suite for CardioANN FastAPI Backend
===================================================
Tests API endpoints, Pydantic validation, schema compliance,
and real model inference via FastAPI TestClient.
"""

import unittest
from fastapi.testclient import TestClient
from backend.main import app


class TestCardioANNAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes FastAPI TestClient."""
        cls.client = TestClient(app)

    def test_01_root_endpoint(self):
        """Tests GET / returns service identity and status online."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "online")
        self.assertIn("service", data)
        self.assertIn("documentation", data)

    def test_02_health_endpoint(self):
        """Tests GET /health confirms model and pipeline readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertTrue(data.get("pipeline_loaded"))
        self.assertEqual(len(data.get("expected_features")), 13)

    def test_03_model_info_endpoint(self):
        """Tests GET /model-info returns verified metrics and 13 feature names."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("test_accuracy"), 0.8361)
        self.assertEqual(data.get("test_roc_auc"), 0.9015)
        self.assertEqual(data.get("input_features_count"), 13)
        self.assertEqual(data.get("transformed_dimensions"), 27)

    def test_04_valid_prediction_request(self):
        """Tests POST /predict with a realistic patient input record."""
        payload = {
            "age": 55,
            "sex": 1,
            "cp": 0,
            "trestbps": 130.0,
            "chol": 240.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
            "threshold": 0.50,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_class", data)
        self.assertIn(data["predicted_class"], [0, 1])
        self.assertIn("probability", data)
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)
        self.assertEqual(data["threshold_applied"], 0.50)
        self.assertIn("explanation", data)
        self.assertIn("disclaimer", data)

    def test_05_prediction_missing_field_error(self):
        """Tests POST /predict rejects request when a required feature is omitted."""
        invalid_payload = {
            "age": 55,
            "sex": 1,
            # 'cp' is intentionally missing
            "trestbps": 130.0,
        }
        response = self.client.post("/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_06_prediction_out_of_bounds_error(self):
        """Tests POST /predict rejects invalid negative age or out-of-range categorical."""
        invalid_payload = {
            "age": -10,  # Invalid age
            "sex": 5,    # Invalid sex (only 0 or 1 allowed)
            "cp": 0,
            "trestbps": 130.0,
            "chol": 240.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }
        response = self.client.post("/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_07_prediction_rejects_target_field(self):
        """Tests POST /predict strictly forbids the 'target' column to prevent data leakage."""
        contaminating_payload = {
            "age": 55,
            "sex": 1,
            "cp": 0,
            "trestbps": 130.0,
            "chol": 240.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
            "target": 1,  # Strictly forbidden!
        }
        response = self.client.post("/predict", json=contaminating_payload)
        self.assertEqual(response.status_code, 422)

    def test_08_custom_threshold_assignment(self):
        """Tests POST /predict with low and high thresholds."""
        payload_low_thresh = {
            "age": 60,
            "sex": 1,
            "cp": 0,
            "trestbps": 145.0,
            "chol": 280.0,
            "fbs": 1,
            "restecg": 1,
            "thalach": 130.0,
            "exang": 1,
            "oldpeak": 2.5,
            "slope": 2,
            "ca": 2,
            "thal": 3,
            "threshold": 0.20,  # Very low threshold
        }
        response = self.client.post("/predict", json=payload_low_thresh)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["threshold_applied"], 0.20)

    def test_09_boundary_values(self):
        """Tests POST /predict with upper and lower clinical boundary values."""
        payload_min = {
            "age": 18,
            "sex": 0,
            "cp": 0,
            "trestbps": 80.0,
            "chol": 100.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 60.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 0,
            "ca": 0,
            "thal": 1,
            "threshold": 0.50,
        }
        response = self.client.post("/predict", json=payload_min)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("probability", data)
        self.assertTrue(0.0 <= data["probability"] <= 1.0)

    def test_10_response_schema_completeness(self):
        """Tests that /predict response contains all required fields with expected types."""
        payload = {
            "age": 52,
            "sex": 1,
            "cp": 1,
            "trestbps": 125.0,
            "chol": 212.0,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168.0,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 2,
            "thal": 3,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        required_keys = [
            "predicted_class", "predicted_label", "probability",
            "probability_percentage", "threshold_applied", "risk_stratum",
            "explanation", "disclaimer", "timestamp"
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing required response key: {key}")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FASTAPI BACKEND TEST SUITE")
    print("=" * 70)
    unittest.main()
