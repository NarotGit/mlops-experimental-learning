"""
src/pipeline/inference.py
Loads the saved sklearn pipeline/files and exposes a predict() function.
Used by the FastAPI app and can also be called standalone.
This module acts as a lightweight bridge rather than heavy train.py.
app.py imports predict() from inference.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
META_PATH = os.path.join(PROJECT_ROOT, "models", "model_meta.json")

_pipeline = None
_meta = None


def _load():
    """Lazy-load model and metadata once."""
    global _pipeline, _meta
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python src/pipeline/train.py"
            )
        _pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)


def get_feature_order() -> list:
    """Returns the ordered list of feature names the model expects."""
    _load()
    return _meta["feature_order"]

def get_model_name() -> str:
    """Returns the name of the best model (for API responses / logs)."""
    _load()
    return _meta["best_model_name"]

def predict(input_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Accepts a dict of raw feature values (as received from JSON),
    returns (prediction: int, confidence: float).
    prediction = 1  → Heart disease present
    prediction = 0  → No heart disease
    confidence      → Probability of the predicted class
    """
    _load()
    feature_order = _meta["feature_order"]
    df = pd.DataFrame([input_data])[feature_order]
    prediction = int(_pipeline.predict(df)[0])
    probabilities = _pipeline.predict_proba(df)[0]
    confidence = float(probabilities[prediction])
    return prediction, confidence


def predict_batch(records: list) -> list:
    """Predict on a batch (list of dicts)."""
    _load()
    feature_order = _meta["feature_order"]
    df = pd.DataFrame(records)[feature_order]
    predictions = _pipeline.predict(df).tolist()
    probabilities = _pipeline.predict_proba(df).tolist()
    results = []
    for pred, probs in zip(predictions, probabilities):
        results.append({
            "prediction": int(pred),
            "confidence": float(probs[pred]),
            "probability_no_disease": float(probs[0]),
            "probability_disease": float(probs[1]),
        })
    return results


if __name__ == "__main__":
    # Quick test
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    pred, conf = predict(sample)
    label = "Heart Disease" if pred == 1 else "No Heart Disease"
    print(f"Model used   : {get_model_name()}")
    print(f"Prediction : {label} ({pred})")
    print(f"Confidence : {conf:.4f}")

    # Batch test
    batch_results = predict_batch([sample, sample])
    print(f"\nBatch test   : {len(batch_results)} results returned")
    print(f"Result[0]    : {batch_results[0]}")