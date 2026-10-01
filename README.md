# Wine MLOps Pipeline

![CI](https://github.com/<your-username>/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

A reproducible, automated MLOps pipeline for classifying wine cultivars
(`sklearn.datasets.load_wine`: 178 samples, 13 features, 3 classes).
The focus is engineering rather than model complexity: modular training code,
Makefile automation, MLflow experiment tracking and model registry, and a
GitHub Actions CI pipeline with a model quality gate.

**Course:** Machine Learning Operations (MLOps), Fall 2026, FAST-NUCES
**Author:** M Saim Ahmad (Roll No: 23F-0018)

---

## Project Structure

```
wine-mlops-pipeline/
├── .github/workflows/ci.yml    # CI: install, lint, test on PR and push to main
├── data/.gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py                 # loading, validation, stratified 80/20 split
│   ├── train.py                # RF + GBM tuning, MLflow logging, registry
│   └── evaluate.py             # loads registered model, scores test split
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # data validation tests
│   └── test_model_gate.py      # MLOps quality gate
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

## Quick Start

Requires Python 3.10 and `make`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                     # upgrade pip, install pinned dependencies
make lint                        # flake8 on src/ and tests/ (max line length 100)
make test                        # pytest, verbose
make train                       # run the MLflow training pipeline
make clean                       # remove bytecode, caches, temp files
```

| Target    | What it does                                                    |
|-----------|-----------------------------------------------------------------|
| `install` | Upgrades pip and installs everything in `requirements.txt`      |
| `lint`    | Runs flake8 over `src/` and `tests/` with `--max-line-length 100` |
| `test`    | Runs all unit tests with `pytest -v`                            |
| `train`   | Runs the training script with default arguments                 |
| `clean`   | Deletes `*.pyc`, `__pycache__`, `.pytest_cache`, temp files     |

## Reproducibility

- Random seed **42** is used for the train/test split, cross-validation
  shuffling, and all model initializations.
- The same Makefile targets run locally and in CI, so results match across
  environments.
- Dependencies are pinned to exact versions in `requirements.txt`.

## Pipeline Overview

### 1. Data (`src/data.py`)
- Loads the Wine dataset.
- Validates there are no null values and exactly 13 features.
- Performs a stratified 80/20 train/test split (`random_state=42`).

### 2. Training and Tuning (`src/train.py`)
Two tree-based model families, each with at least three hyperparameter
configurations (6 runs in total):

- **Family A:** `RandomForestClassifier`
- **Family B:** `GradientBoostingClassifier`

Each configuration is evaluated with **5-fold stratified cross-validation**
on the training split, recording train and validation **Macro F1**,
**Accuracy**, and **Log Loss**.

### 3. MLflow Tracking and Registry
- Local file-based/SQLite backend, experiment `Wine-Cultivar-Classification`.
- One MLflow run per configuration, logging parameters, metrics and tags.
- Model signature inferred with `mlflow.models.infer_signature`, an input
  example saved, and the model logged with `mlflow.sklearn.log_model()`.
- The run with the best **validation macro F1** is registered as
  **`WineClassifier`** and given the alias **`champion`**.

View the results locally:

```bash
mlflow ui
# open http://127.0.0.1:5000
```

### 4. Inference Check (`src/evaluate.py`)
Loads `models:/WineClassifier@champion` and computes final metrics on the
held-out test split.

## Results

| Run | Model Family     | Val Accuracy | Val Macro F1 | Val Log Loss |
|-----|------------------|--------------|--------------|--------------|
| RandomForest-cfg2 (champion) | RandomForest | 0.9791 | 0.9789 | 0.1630 |
| RandomForest-cfg3 | RandomForest     | 0.9791 | 0.9789 | 0.1630 |
| RandomForest-cfg1 | RandomForest     | 0.9653 | 0.9665 | 0.2055 |
| GradientBoosting-cfg1 | GradientBoosting | 0.9581 | 0.9593 | 0.1477 |
| GradientBoosting-cfg2 | GradientBoosting | 0.9163 | 0.9181 | 0.2565 |
| GradientBoosting-cfg3 | GradientBoosting | 0.9094 | 0.9112 | 0.4347 |

## CI/CD

The workflow in `.github/workflows/ci.yml`:

- Triggers on `pull_request` to `main` and `push` to `main`.
- Runs on `ubuntu-latest` with Python 3.10.
- Steps: checkout, set up Python, `make install`, `make lint`, `make test`.

### Model Quality Gate (`tests/test_model_gate.py`)

A change fails CI if any of these checks fail:

| Gate                  | Requirement                                  |
|-----------------------|----------------------------------------------|
| Metric threshold      | Validation Macro F1 >= 0.88                  |
| Inference latency     | Batch inference time <= 30 ms                |
| Output schema         | Predictions are class indices 0, 1 or 2 only |

## Git Workflow

- `main` is kept clean; work happens on feature branches
  (for example `feature/mlflow-tracking`) and is merged through pull requests.
- A deliberate merge conflict was created on the `conflict-simulation` branch
  (conflicting edit to the same configuration line on `main`), then resolved
  manually and committed. Terminal logs are included in the report.

View the history:

```bash
git log --graph --oneline --all
```

## Ignored Files

`.gitignore` excludes `mlruns/`, `.pytest_cache/`, `__pycache__/`, `.venv/`
and transient model artifacts.

## Report

The submission report is `MLOps_A01_RollNumber.pdf`. It contains the
hyperparameter table, MLflow UI screenshots, CI evidence, the Git tree log
and a short analysis.
