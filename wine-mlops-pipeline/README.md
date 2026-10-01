# Wine Cultivar Classification: End-to-End MLOps Pipeline

[![CI Pipeline](https://github.com/USERNAME/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/wine-mlops-pipeline/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, reproducible Continuous Integration and Continuous Deployment (CI/CD) MLOps pipeline for multi-class chemical cultivar classification using `sklearn.datasets.load_wine` (178 samples, 13 features, 3 target classes).

---

## 📌 Repository Architecture

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow
├── data/
│   └── .gitkeep               # Directory placeholder for raw/processed datasets
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data.py                # Data loading, validation, and stratified 80/20 splitting
│   ├── train.py               # 5-fold CV training, MLflow tracking, and model registry
│   └── evaluate.py            # Inference verification using champion model alias
├── tests/
│   ├── __init__.py            # Test package initialization
│   ├── test_data.py           # Unit tests for data schema and validation checks
│   └── test_model_gate.py     # Automated MLOps Quality Gate (F1, latency, schema)
├── .gitignore                 # Excludes caches, virtual environments, and MLflow db
├── Makefile                   # Automation entry points (install, lint, test, train, clean)
├── requirements.txt           # Pinned dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/USERNAME/wine-mlops-pipeline.git
cd wine-mlops-pipeline

# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate
# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Makefile Automation Targets

All development workflows are driven through the GNU `Makefile`:

| Target | Command | Description |
| :--- | :--- | :--- |
| `install` | `make install` | Upgrades `pip` and installs pinned dependencies from `requirements.txt` |
| `lint` | `make lint` | Runs `flake8` over `src/` and `tests/` with maximum line length 100 |
| `test` | `make test` | Executes all unit tests and MLOps Quality Gates with `pytest -v` |
| `train` | `make train` | Runs dual classifier 5-fold CV and registers champion in MLflow |
| `clean` | `make clean` | Removes bytecode (`*.pyc`), `__pycache__`, and `.pytest_cache` |

---

## 🔬 MLflow Tracking & Model Registry

The training script `src/train.py` systematically trains two classifier families:
- **Family A**: `RandomForestClassifier` (3 distinct configurations)
- **Family B**: `GradientBoostingClassifier` (3 distinct configurations)

### Experiment Execution
```bash
make train
```

During training, each run logs:
- **Hyperparameters**: `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, `random_state=42`.
- **Metrics**: 5-fold cross-validated mean `val_macro_f1`, `val_accuracy`, `val_log_loss`, `train_macro_f1`, `train_accuracy`, `train_log_loss`.
- **Artifacts**: Input example, model signature schema, and serialized Scikit-learn model.
- **Model Registry Promotion**: Compares all runs, automatically registers the top-performing model as `WineClassifier`, and assigns the `champion` alias.

### Launching the MLflow UI
To inspect experiment runs, compare metric curves, and view the Model Registry:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` in your web browser.

### Inference Verification
Test the registered champion model on the unseen test split:
```bash
python -m src.evaluate
```

---

## 🛡️ Automated MLOps Quality Gate

Located in `tests/test_model_gate.py`, this quality gate ensures that sub-standard or regression models never enter production:

1. **Metric Threshold Gate**: Champion model validation Macro F1 score must be $\ge 0.88$.
2. **Inference Latency Gate**: Batch inference latency for the test split must be $\le 30\text{ ms}$.
3. **Output Schema Integrity**: Predicted classes must strictly be integers within $\{0, 1, 2\}$.

---

## 🔄 CI/CD Pipeline (GitHub Actions)

The workflow defined in `.github/workflows/ci.yml` is automatically triggered on:
- Every `push` to `main`.
- Every `pull_request` targeting `main`.

It executes in an `ubuntu-latest` environment with Python 3.10 and validates the entire pipeline via `make install`, `make lint`, and `make test`.
