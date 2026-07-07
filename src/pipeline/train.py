"""
src/pipeline/train.py
Trains Logistic Regression and Random Forest classifiers on the Heart Disease
UCI dataset, tracks all experiments with MLflow, and saves the best model.

Run from project root:
    python src/pipeline/train.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)

import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "heart.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run: python data/download_data.py"
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Preprocessing pipeline builder
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# Plotting helpers (saved as artifacts)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_roc_curve(fpr, tpr, auc_score, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Train + evaluate one model (one MLflow run)
# ---------------------------------------------------------------------------
def train_and_log(
    model_name: str,
    classifier,
    param_grid: dict,
    X_train, X_test, y_train, y_test,
    experiment_name: str,
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):

        # -- Build full pipeline --
        preprocessor = build_preprocessor()
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        # -- Hyperparameter tuning --
        print(f"\n[{model_name}] Running GridSearchCV...")
        grid_search = GridSearchCV(
            full_pipeline, param_grid,
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # -- Log params --
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        mlflow.log_param("model_name", model_name)

        # -- Cross-val on training set --
        cv_auc = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", float(cv_auc.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_auc.std()))

        # -- Test set evaluation --
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy":  accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall":    recall_score(y_test, y_pred),
            "test_f1":        f1_score(y_test, y_pred),
            "test_roc_auc":   roc_auc_score(y_test, y_prob),
        }
        mlflow.log_metrics(metrics)

        print(f"[{model_name}] Best params: {best_params}")
        print(f"[{model_name}] Test metrics: {metrics}")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

        # -- Confusion matrix plot --
        cm = confusion_matrix(y_test, y_pred)
        cm_path = f"/tmp/cm_{model_name.replace(' ', '_')}.png"
        plot_confusion_matrix(cm, ["No Disease", "Disease"],
                              f"Confusion Matrix – {model_name}", cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # -- ROC curve plot --
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_path = f"/tmp/roc_{model_name.replace(' ', '_')}.png"
        plot_roc_curve(fpr, tpr, metrics["test_roc_auc"],
                       f"ROC Curve – {model_name}", roc_path)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # -- Log sklearn model --
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"[{model_name}] MLflow run_id: {run_id}")

    return best_pipeline, metrics, run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Load data ----
    df = load_data(DATA_PATH)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    experiment_name = "heart_disease_classification"

    # ---- Model 1: Logistic Regression ----
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs"],
        "classifier__max_iter": [1000],
    }
    lr_pipeline, lr_metrics, lr_run_id = train_and_log(
        model_name="Logistic Regression",
        classifier=LogisticRegression(random_state=42),
        param_grid=lr_param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        experiment_name=experiment_name,
    )

    # ---- Model 2: Random Forest ----
    rf_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    }
    rf_pipeline, rf_metrics, rf_run_id = train_and_log(
        model_name="Random Forest",
        classifier=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        experiment_name=experiment_name,
    )

    # ---- Pick best model by ROC-AUC ----
    if rf_metrics["test_roc_auc"] >= lr_metrics["test_roc_auc"]:
        best_pipeline, best_name = rf_pipeline, "Random Forest"
    else:
        best_pipeline, best_name = lr_pipeline, "Logistic Regression"

    # ---- Save best model to disk ----
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model: {best_name}")
    print(f"Model saved to {model_path}")

    # ---- Save metadata for API to read ----
    meta = {
        "best_model_name": best_name,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model metadata saved to {meta_path}")

    # ---- Comparison summary ----
    print("\n=== Model Comparison ===")
    print(f"{'Metric':<20} {'Logistic Reg':>15} {'Random Forest':>15}")
    print("-" * 52)
    for k in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
        print(f"{k:<20} {lr_metrics[k]:>15.4f} {rf_metrics[k]:>15.4f}")


if __name__ == "__main__":
    main()