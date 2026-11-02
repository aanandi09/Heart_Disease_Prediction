"""
Automated Test Suite for Heart Disease Prediction using ANN
============================================================
Validates dataset integrity, preprocessing pipeline, neural network inference,
probability calibration, edge cases, and generated evaluation artifacts.
"""

import unittest
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Set base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

from src.preprocess import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    load_raw_data,
    build_preprocessor,
)


class TestHeartDiseasePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads dataset and serialized model artifacts once for all tests."""
        cls.dataset_path = BASE_DIR / "dataset" / "heart.csv"
        cls.model_path = BASE_DIR / "models" / "heart_disease_ann.keras"
        cls.pipeline_path = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
        cls.results_dir = BASE_DIR / "results"

        # Assert files exist
        assert cls.dataset_path.exists(), f"Dataset file missing at {cls.dataset_path}"
        assert cls.model_path.exists(), f"Model file missing at {cls.model_path}"
        assert cls.pipeline_path.exists(), f"Pipeline file missing at {cls.pipeline_path}"

        cls.df_raw = pd.read_csv(cls.dataset_path)
        cls.model = keras.models.load_model(cls.model_path)
        cls.pipeline = joblib.load(cls.pipeline_path)

    def test_01_dataset_schema_and_types(self):
        """Checks dataset column presence, row count, and absence of missing values."""
        self.assertEqual(len(self.df_raw), 1025, "Dataset row count should be exactly 1025.")
        expected_columns = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
        for col in expected_columns:
            self.assertIn(col, self.df_raw.columns, f"Mandatory column '{col}' missing from dataset.")

        # Zero missing values check
        null_counts = self.df_raw.isnull().sum().sum()
        self.assertEqual(null_counts, 0, "Dataset should contain zero missing values.")

        # Target column binary validation
        unique_targets = set(self.df_raw[TARGET_COLUMN].unique())
        self.assertEqual(unique_targets, {0, 1}, "Target variable must be binary {0, 1}.")

    def test_02_preprocessing_transformation_shape(self):
        """Verifies ColumnTransformer output dimensions (27 engineered features)."""
        sample_batch = self.df_raw[ALL_FEATURE_COLUMNS].head(10)
        transformed = self.pipeline.transform(sample_batch)

        self.assertEqual(transformed.shape[0], 10, "Batch size must be preserved through pipeline.")
        self.assertEqual(transformed.shape[1], 27, "Engineered feature matrix must contain exactly 27 columns.")
        self.assertFalse(np.isnan(transformed).any(), "Preprocessed matrix contains NaN values.")
        self.assertFalse(np.isinf(transformed).any(), "Preprocessed matrix contains Inf values.")

    def test_03_ann_inference_and_probability_bounds(self):
        """Checks neural network inference, output shape, and valid probability ranges [0, 1]."""
        sample_batch = self.df_raw[ALL_FEATURE_COLUMNS].head(25)
        X_proc = self.pipeline.transform(sample_batch)
        probabilities = self.model.predict(X_proc, verbose=0).ravel()

        self.assertEqual(len(probabilities), 25, "Inference output length must match batch input length.")
        self.assertTrue(np.all(probabilities >= 0.0), "Predicted probabilities cannot be negative.")
        self.assertTrue(np.all(probabilities <= 1.0), "Predicted probabilities cannot exceed 1.0.")
        self.assertTrue(np.all(np.isfinite(probabilities)), "All predicted probabilities must be finite numbers.")

    def test_04_single_patient_prediction(self):
        """Verifies prediction on a single arbitrary clinical input record."""
        single_patient = pd.DataFrame(
            [
                {
                    "age": 55,
                    "sex": 1,
                    "cp": 2,
                    "trestbps": 130,
                    "chol": 240,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 1.2,
                    "slope": 1,
                    "ca": 0,
                    "thal": 2,
                }
            ]
        )
        X_proc = self.pipeline.transform(single_patient)
        prob = float(self.model.predict(X_proc, verbose=0)[0][0])

        self.assertIsInstance(prob, float, "Model prediction should return a scalar float.")
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_05_error_handling_missing_column(self):
        """Verifies that the pipeline raises an informative ValueError if a required column is missing."""
        invalid_input = pd.DataFrame(
            [
                {
                    "age": 50,
                    "sex": 1,
                    # 'cp' is intentionally missing
                    "trestbps": 120,
                    "chol": 200,
                }
            ]
        )
        with self.assertRaises(Exception):
            self.pipeline.transform(invalid_input)

    def test_06_evaluation_artifacts_exist(self):
        """Confirms all generated diagnostic visual artifacts and JSON files exist on disk."""
        required_artifacts = [
            "final_metrics.json",
            "classification_report.txt",
            "model_comparison.csv",
            "confusion_matrix.png",
            "roc_curve.png",
            "training_history.png",
            "feature_analysis.png",
        ]
        for art in required_artifacts:
            art_path = self.results_dir / art
            self.assertTrue(art_path.exists(), f"Evaluation artifact '{art}' missing from results directory.")
            self.assertGreater(art_path.stat().st_size, 0, f"Artifact '{art}' is empty.")

    def test_07_verified_metrics_consistency(self):
        """Validates that final_metrics.json contains honest, verified test metrics."""
        json_path = self.results_dir / "final_metrics.json"
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertIn("accuracy", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Check empirical zero-leakage range
        self.assertAlmostEqual(metrics["accuracy"], 0.8361, places=3)
        self.assertAlmostEqual(metrics["roc_auc"], 0.9015, places=3)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUTOMATED REGRESSION & QUALITY ASSURANCE TEST SUITE")
    print("=" * 70)
    unittest.main()
