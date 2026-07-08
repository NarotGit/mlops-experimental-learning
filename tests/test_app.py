"""
tests/test_all.py
Pytest test suite for:
  - Data loading and validation
  - Preprocessing pipeline
  - Model prediction
  - FastAPI /predict and /health endpoints

Run: pytest tests/ -v --cov=src --cov-report=term-missing
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# =========================================================================
# Fixtures
# =========================================================================

SAMPLE_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}

SAMPLE_INPUT_NO_DISEASE = {
    "age": 35, "sex": 0, "cp": 0, "trestbps": 120,
    "chol": 180, "fbs": 0, "restecg": 0, "thalach": 175,
    "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2,
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")


# Data tests
class TestData:
    """does heart.csv exist? right columns? binary target?
        (catches id data is broken early)"""
    def test_dataset_file_exists(self):
        assert os.path.exists(DATA_PATH), (
            f"Dataset not found at {DATA_PATH}. Run: python data/download_data.py"
        )

    def test_dataset_loads(self):
        df = pd.read_csv(DATA_PATH)
        assert len(df) > 0, "Dataset is empty"

    def test_dataset_has_required_columns(self):
        df = pd.read_csv(DATA_PATH)
        required = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "target",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_is_binary(self):
        df = pd.read_csv(DATA_PATH)
        assert set(df["target"].unique()).issubset({0, 1}), (
            "Target column should only contain 0 and 1"
        )

    def test_dataset_has_minimum_rows(self):
        df = pd.read_csv(DATA_PATH)
        assert len(df) >= 100, f"Expected at least 100 rows, got {len(df)}"

    def test_no_all_null_columns(self):
        df = pd.read_csv(DATA_PATH)
        for col in df.columns:
            assert df[col].notna().any(), f"Column {col} is entirely null"


# Preprocessing pipeline tests
class TestPreprocessor:
    """checks does ColumnTransformer work?
       no NaNs after imputation? right output shape?
      (catches pipeline config bugs)"""
    def test_build_preprocessor(self):
        from src.pipeline.train import build_preprocessor
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_preprocessor_fit_transform(self):
        from src.pipeline.train import build_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
        df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
        if df is None:
            pytest.skip("Dataset not available")
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(X), "Row count mismatch after preprocessing"
        assert X_transformed.shape[1] > len(NUMERICAL_FEATURES), (
            "Expected more columns after one-hot encoding"
        )
        assert not np.any(np.isnan(X_transformed)), "NaN values found after preprocessing"


# Model and inference tests
class TestModel:
    """does predict() return int + float?
       is prediction 0 or 1? confidence in [0,1]?
       (catches model file corruption or inference bugs)"""
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), (
            f"Model not found at {MODEL_PATH}. Run: python src/pipeline/train.py"
        )

    def test_predict_returns_valid_types(self):
        from src.pipeline.inference import predict
        pred, conf = predict(SAMPLE_INPUT)
        assert isinstance(pred, int), "Prediction should be int"
        assert isinstance(conf, float), "Confidence should be float"

    def test_predict_binary_output(self):
        from src.pipeline.inference import predict
        pred, conf = predict(SAMPLE_INPUT)
        assert pred in [0, 1], f"Prediction must be 0 or 1, got {pred}"

    def test_confidence_in_range(self):
        from src.pipeline.inference import predict
        pred, conf = predict(SAMPLE_INPUT)
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1] range"

    def test_predict_batch_length(self):
        from src.pipeline.inference import predict_batch
        results = predict_batch([SAMPLE_INPUT, SAMPLE_INPUT_NO_DISEASE])
        assert len(results) == 2, "Batch output should have same length as input"

    def test_predict_batch_structure(self):
        from src.pipeline.inference import predict_batch
        results = predict_batch([SAMPLE_INPUT])
        r = results[0]
        assert "prediction" in r
        assert "confidence" in r
        assert "probability_no_disease" in r
        assert "probability_disease" in r
        assert abs(r["probability_no_disease"] + r["probability_disease"] - 1.0) < 1e-5

    def test_feature_order_consistent(self):
        from src.pipeline.inference import get_feature_order
        order = get_feature_order()
        assert isinstance(order, list)
        assert len(order) == 13, f"Expected 13 features, got {len(order)}"


# API tests (TestClient — no server needed)
@pytest.fixture(scope="module")
def client():
    from src.api.app import app
    return TestClient(app)


class TestAPI:
    """does /predict return 200?
    does /health say "healthy"?
    does bad input get rejected with 422?
    (catches any API contract bugs before dockerization)"""
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_valid_input(self, client):
        response = client.post("/predict", json=SAMPLE_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_label_matches_prediction(self, client):
        response = client.post("/predict", json=SAMPLE_INPUT)
        data = response.json()
        if data["prediction"] == 1:
            assert data["label"] == "Heart Disease"
        else:
            assert data["label"] == "No Heart Disease"

    def test_predict_missing_field_returns_422(self, client):
        bad_input = {k: v for k, v in SAMPLE_INPUT.items() if k != "age"}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_predict_invalid_sex_value(self, client):
        bad_input = {**SAMPLE_INPUT, "sex": 5}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_metrics_endpoint_reachable(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200