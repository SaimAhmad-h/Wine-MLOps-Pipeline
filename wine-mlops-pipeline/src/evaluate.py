"""Inference verification module to evaluate the champion model on the test split."""

import argparse
from typing import Dict
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

from src.data import load_and_validate_data

REGISTERED_MODEL_NAME = "WineClassifier"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def evaluate_champion_model(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = "champion",
) -> Dict[str, float]:
    """Load the champion model from the MLflow registry and compute test split metrics.

    Args:
        tracking_uri (str, optional): MLflow tracking URI. Defaults to DEFAULT_TRACKING_URI.
        model_name (str, optional): Registered model name. Defaults to REGISTERED_MODEL_NAME.
        alias (str, optional): Model alias. Defaults to "champion".

    Returns:
        Dict[str, float]: Final evaluation metrics on the test dataset.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{alias}"

    print(f"Loading champion model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    _, x_test, _, y_test = load_and_validate_data(random_state=42)

    # Perform inference
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    # Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    loss = float(log_loss(y_test, y_prob, labels=[0, 1, 2]))

    print("=" * 60)
    print("CHAMPION MODEL TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model URI:       {model_uri}")
    print(f"Test Samples:    {len(y_test)}")
    print(f"Test Accuracy:   {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Test Macro F1:   {macro_f1:.4f}")
    print(f"Test Log Loss:   {loss:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1", "Class 2"]))
    print("=" * 60)

    return {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_log_loss": loss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Champion Model on Test Split")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    args = parser.parse_args()
    evaluate_champion_model(tracking_uri=args.tracking_uri)
