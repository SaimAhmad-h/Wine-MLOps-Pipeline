"""Data ingestion and validation module for the Wine Cultivar classification pipeline."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

EXPECTED_FEATURE_COUNT = 13
EXPECTED_SAMPLE_COUNT = 178
EXPECTED_CLASSES = {0, 1, 2}


def get_raw_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the raw Wine dataset from scikit-learn.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target labels y.
    """
    wine_dataset = load_wine(as_frame=True)
    features_df = wine_dataset.data
    target_series = wine_dataset.target
    return features_df, target_series


def validate_data(features_df: pd.DataFrame, target_series: pd.Series) -> None:
    """Validate data integrity, schema, and absence of null values.

    Args:
        features_df (pd.DataFrame): Input feature matrix.
        target_series (pd.Series): Target class labels.

    Raises:
        ValueError: If null values exist, feature count != 13, or sample count is invalid.
    """
    if features_df is None or target_series is None:
        raise ValueError("Dataframe or target series cannot be None.")

    if len(features_df) != len(target_series):
        raise ValueError(
            f"Mismatched sample counts: features has {len(features_df)}, "
            f"target has {len(target_series)}."
        )

    if features_df.isnull().sum().sum() > 0:
        raise ValueError("Data validation failed: Null values detected in features.")

    if target_series.isnull().sum() > 0:
        raise ValueError("Data validation failed: Null values detected in target series.")

    if features_df.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Data validation failed: Expected exactly {EXPECTED_FEATURE_COUNT} features, "
            f"but found {features_df.shape[1]}."
        )

    unique_classes = set(target_series.unique())
    if not unique_classes.issubset(EXPECTED_CLASSES):
        raise ValueError(
            f"Data validation failed: Unexpected class labels {unique_classes}. "
            f"Expected subset of {EXPECTED_CLASSES}."
        )


def load_and_validate_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load Wine dataset, validate integrity, and perform stratified 80/20 train-test split.

    Args:
        test_size (float, optional): Proportion of test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    features_df, target_series = get_raw_data()
    validate_data(features_df, target_series)

    x_train, x_test, y_train, y_test = train_test_split(
        features_df,
        target_series,
        test_size=test_size,
        stratify=target_series,
        random_state=random_state,
    )

    # Post-split verification
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_tr, x_te, y_tr, y_te = load_and_validate_data()
    print("Data successfully loaded and validated:")
    print(f"  Training set: {x_tr.shape[0]} samples, {x_tr.shape[1]} features")
    print(f"  Testing set:  {x_te.shape[0]} samples, {x_te.shape[1]} features")
