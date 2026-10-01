"""Model training, 5-fold cross-validation, and MLflow experiment tracking pipeline."""

import argparse
from typing import Any, Dict, List, Tuple
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.data import load_and_validate_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def get_hyperparameter_grids() -> List[Dict[str, Any]]:
    """Define candidate hyperparameter configurations for RF and GBM classifiers.

    Returns:
        List[Dict[str, Any]]: List of dictionary configurations.
    """
    configs = [
        # Random Forest configurations
        {
            "family": "RandomForestClassifier",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 2,
                "random_state": 42,
            },
        },
        {
            "family": "RandomForestClassifier",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 4,
                "random_state": 42,
            },
        },
        {
            "family": "RandomForestClassifier",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 150,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 42,
            },
        },
        # Gradient Boosting configurations
        {
            "family": "GradientBoostingClassifier",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 50,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
            },
        },
        {
            "family": "GradientBoostingClassifier",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
        },
        {
            "family": "GradientBoostingClassifier",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 150,
                "learning_rate": 0.2,
                "max_depth": 4,
                "random_state": 42,
            },
        },
    ]
    return configs


def evaluate_cv(
    estimator, x_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5
) -> Dict[str, float]:
    """Evaluate an estimator using Stratified K-Fold cross validation on the training set.

    Args:
        estimator: Scikit-learn estimator instance.
        x_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        n_splits (int, optional): Number of folds. Defaults to 5.

    Returns:
        Dict[str, float]: Aggregated mean metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for fold_train_idx, fold_val_idx in skf.split(x_train, y_train):
        fold_x_tr = x_train.iloc[fold_train_idx]
        fold_y_tr = y_train.iloc[fold_train_idx]
        fold_x_val = x_train.iloc[fold_val_idx]
        fold_y_val = y_train.iloc[fold_val_idx]

        model = clone(estimator)
        model.fit(fold_x_tr, fold_y_tr)

        # Train fold predictions
        y_pred_tr = model.predict(fold_x_tr)
        y_prob_tr = model.predict_proba(fold_x_tr)
        train_f1s.append(f1_score(fold_y_tr, y_pred_tr, average="macro"))
        train_accs.append(accuracy_score(fold_y_tr, y_pred_tr))
        train_losses.append(log_loss(fold_y_tr, y_prob_tr, labels=[0, 1, 2]))

        # Validation fold predictions
        y_pred_val = model.predict(fold_x_val)
        y_prob_val = model.predict_proba(fold_x_val)
        val_f1s.append(f1_score(fold_y_val, y_pred_val, average="macro"))
        val_accs.append(accuracy_score(fold_y_val, y_pred_val))
        val_losses.append(log_loss(fold_y_val, y_prob_val, labels=[0, 1, 2]))

    return {
        "train_macro_f1": float(np.mean(train_f1s)),
        "val_macro_f1": float(np.mean(val_f1s)),
        "train_accuracy": float(np.mean(train_accs)),
        "val_accuracy": float(np.mean(val_accs)),
        "train_log_loss": float(np.mean(train_losses)),
        "val_log_loss": float(np.mean(val_losses)),
    }


def run_training(
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> Tuple[str, str, float]:
    """Execute training pipeline across all candidate hyperparameter configurations.

    Args:
        tracking_uri (str, optional): MLflow tracking URI. Defaults to DEFAULT_TRACKING_URI.

    Returns:
        Tuple[str, str, float]: Best run ID, best model family, best validation macro F1 score.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, _, y_train, _ = load_and_validate_data(random_state=42)
    configs = get_hyperparameter_grids()

    best_run_id = ""
    best_family = ""
    best_val_f1 = -1.0

    print(f"Starting MLflow experiment: {EXPERIMENT_NAME}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Total hyperparameter configurations to test: {len(configs)}")
    print("-" * 80)

    for i, cfg in enumerate(configs, 1):
        family = cfg["family"]
        model_cls = cfg["model_class"]
        params = cfg["params"]

        run_name = f"{family}_config_{i}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Model evaluation using 5-fold CV
            estimator = model_cls(**params)
            metrics = evaluate_cv(estimator, x_train, y_train, n_splits=5)

            # Fit model on full training split for artifact packaging
            full_model = model_cls(**params)
            full_model.fit(x_train, y_train)

            # Infer model signature & extract input example
            input_example = x_train.iloc[:5]
            predictions_example = full_model.predict(input_example)
            signature = infer_signature(input_example, predictions_example)

            # Log parameters, metrics, tags
            clean_params = {k: ("None" if v is None else v) for k, v in params.items()}
            mlflow.log_params(clean_params)
            mlflow.log_param("model_family", family)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "model_family": family,
                    "cv_folds": 5,
                    "dataset": "wine_cultivars",
                    "seed": 42,
                }
            )

            # Log model artifact with signature and input example
            mlflow.sklearn.log_model(
                sk_model=full_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            print(
                f"[{i}/{len(configs)}] {family:<26} | "
                f"Val F1: {metrics['val_macro_f1']:.4f} | "
                f"Val Acc: {metrics['val_accuracy']:.4f} | "
                f"Val Loss: {metrics['val_log_loss']:.4f} | "
                f"Run ID: {run_id[:8]}"
            )

            if metrics["val_macro_f1"] > best_val_f1:
                best_val_f1 = metrics["val_macro_f1"]
                best_run_id = run_id
                best_family = family

    print("-" * 80)
    print(f"Top Performer: {best_family} (Run ID: {best_run_id})")
    print(f"Highest Validation Macro F1: {best_val_f1:.4f}")

    # Register champion model
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Registering model from URI: {model_uri} as '{REGISTERED_MODEL_NAME}'...")

    registered_version = mlflow.register_model(
        model_uri=model_uri, name=REGISTERED_MODEL_NAME
    )

    print(
        f"Promoting version {registered_version.version} of '{REGISTERED_MODEL_NAME}' "
        f"with alias 'champion'..."
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_version.version,
    )
    print(
        f"Successfully promoted version {registered_version.version} to champion alias!"
    )

    return best_run_id, best_family, best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cultivar Classifiers and Track with MLflow")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    args = parser.parse_args()
    run_training(tracking_uri=args.tracking_uri)
