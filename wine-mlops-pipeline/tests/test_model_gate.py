"""Automated MLOps Quality Gate tests.

Enforces:
1. Metric Threshold Gate: Validation Macro F1-score >= 0.88.
2. Inference Latency Gate: Batch inference latency <= 30 ms.
3. Output Schema Integrity: Class indices strictly within {0, 1, 2}.
"""

import time
from typing import Tuple
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pytest

from src.data import EXPECTED_CLASSES, load_and_validate_data
from src.train import (
    DEFAULT_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    run_training,
)

METRIC_F1_THRESHOLD = 0.88
MAX_LATENCY_SECONDS = 0.030  # 30 milliseconds


@pytest.fixture(scope="session")
def champion_model_info() -> Tuple[object, float]:
    """Retrieve or train the champion model and its validation Macro F1 metric."""
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    client = MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME, alias="champion"
        )
        run_id = model_version.run_id
        run = client.get_run(run_id)
        val_macro_f1 = run.data.metrics.get("val_macro_f1")
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    except Exception:
        # If not already trained, run training pipeline to ensure self-contained tests
        best_run_id, _, best_val_f1 = run_training(DEFAULT_TRACKING_URI)
        val_macro_f1 = best_val_f1
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")

    return model, val_macro_f1


def test_metric_threshold_gate(champion_model_info):
    """Quality Gate 1: Ensure champion validation Macro F1 score meets or exceeds 0.88."""
    _, val_macro_f1 = champion_model_info
    assert val_macro_f1 is not None, "Champion model must have a recorded val_macro_f1 metric."
    assert (
        val_macro_f1 >= METRIC_F1_THRESHOLD
    ), f"Quality Gate Failed: Validation Macro F1 {val_macro_f1:.4f} < {METRIC_F1_THRESHOLD}"


def test_inference_latency_gate(champion_model_info):
    """Quality Gate 2: Ensure batch inference latency on test split is <= 30 ms."""
    model, _ = champion_model_info
    _, x_test, _, _ = load_and_validate_data(random_state=42)

    # Warm-up run to eliminate cold-start artifact loading overhead
    _ = model.predict(x_test)

    # Latency measurement over multiple iterations for stability
    iterations = 20
    latencies = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = model.predict(x_test)
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

    mean_latency = float(np.mean(latencies))
    print(f"\nAverage batch inference time for {len(x_test)} samples: {mean_latency * 1000:.3f} ms")

    assert (
        mean_latency <= MAX_LATENCY_SECONDS
    ), f"Quality Gate Failed: Batch inference latency {mean_latency * 1000:.2f} ms > 30 ms."


def test_output_schema_integrity(champion_model_info):
    """Quality Gate 3: Ensure predictions only produce valid class labels {0, 1, 2}."""
    model, _ = champion_model_info
    _, x_test, _, _ = load_and_validate_data(random_state=42)

    predictions = model.predict(x_test)

    # Check output shape
    assert predictions.shape == (len(x_test),), "Output prediction shape must match sample count."

    # Check class indices
    unique_preds = set(np.unique(predictions))
    assert unique_preds.issubset(
        EXPECTED_CLASSES
    ), f"Quality Gate Failed: Model predicted invalid classes {unique_preds - EXPECTED_CLASSES}."

    # Ensure integer indices
    assert np.issubdtype(
        predictions.dtype, np.integer
    ), f"Predicted class labels must be integers, got {predictions.dtype}."
