"""Unit tests for data ingestion, schema validation, and stratified splitting."""

import pytest

from src.data import (
    EXPECTED_CLASSES,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    get_raw_data,
    load_and_validate_data,
    validate_data,
)


def test_raw_data_shape_and_schema():
    """Verify raw Wine dataset has expected shape and sample dimensions."""
    x, y = get_raw_data()
    assert len(x) == EXPECTED_SAMPLE_COUNT
    assert x.shape[1] == EXPECTED_FEATURE_COUNT
    assert len(y) == EXPECTED_SAMPLE_COUNT


def test_data_validation_clean_pass():
    """Verify valid dataset passes validation without error."""
    x, y = get_raw_data()
    # Should not raise any exception
    validate_data(x, y)


def test_validation_catches_null_values():
    """Verify validation triggers error when null values are introduced."""
    x, y = get_raw_data()
    x_corrupt = x.copy()
    x_corrupt.iloc[0, 0] = None

    with pytest.raises(ValueError, match="Null values detected in features"):
        validate_data(x_corrupt, y)

    y_corrupt = y.copy()
    y_corrupt.iloc[0] = None
    with pytest.raises(ValueError, match="Null values detected in target series"):
        validate_data(x, y_corrupt)


def test_validation_catches_incorrect_feature_count():
    """Verify validation triggers error when feature count is not 13."""
    x, y = get_raw_data()
    x_dropped = x.drop(columns=[x.columns[0]])

    with pytest.raises(ValueError, match="Expected exactly 13 features"):
        validate_data(x_dropped, y)


def test_validation_catches_invalid_classes():
    """Verify validation triggers error when unexpected class labels are present."""
    x, y = get_raw_data()
    y_corrupt = y.copy()
    y_corrupt.iloc[0] = 99

    with pytest.raises(ValueError, match="Unexpected class labels"):
        validate_data(x, y_corrupt)


def test_train_test_split_sizes_and_features():
    """Verify stratified split yields 80/20 proportion and maintains feature count."""
    x_train, x_test, y_train, y_test = load_and_validate_data(
        test_size=0.2, random_state=42
    )

    # 178 * 0.8 = 142.4 -> 142 samples
    assert len(x_train) == 142
    # 178 * 0.2 = 35.6 -> 36 samples
    assert len(x_test) == 36

    assert x_train.shape[1] == EXPECTED_FEATURE_COUNT
    assert x_test.shape[1] == EXPECTED_FEATURE_COUNT

    # Ensure all target classes belong to {0, 1, 2}
    assert set(y_train.unique()).issubset(EXPECTED_CLASSES)
    assert set(y_test.unique()).issubset(EXPECTED_CLASSES)


def test_stratification_distribution():
    """Verify that class balance is preserved between train and test splits."""
    _, _, y_train, y_test = load_and_validate_data(test_size=0.2, random_state=42)

    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()

    # The proportion difference for any class between splits should be minimal (< 0.05)
    for c in EXPECTED_CLASSES:
        assert abs(train_dist[c] - test_dist[c]) < 0.05
