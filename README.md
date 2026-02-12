# School Dropout Prediction in Argentina

## Overview

This project develops an **Early Warning System (EWS)** to identify at-risk students prone to school dropout in Argentina, using data from the Permanent Household Survey (EPH - *Encuesta Permanente de Hogares*).

### Context

Education in Argentina is guaranteed by the National Constitution and regulated by Law No. 26,206 (National Education Law). Compulsory education spans 14 consecutive years, from preschool through secondary school. While primary education achieves near-universal coverage (99% attendance), secondary school completion remains a challenge—approximately 20% of young adults aged 18-24 have not completed compulsory education.

The dropout rate at the primary level is low (0.41%), but increases significantly at secondary level (7.66%), affecting 16.8% of students in their final year of compulsory education.

### Project Goals

1. Develop a predictive model for student dropout to power the EWS.
2. Build an interactive dashboard for analyzing educational data from the EPH.
3. Integrate the EWS into the analytics dashboard for actionable insights.

### Improvements Over Prior Work

This project improves upon previous research (Michel Torino's thesis) in several key ways:

- **Longitudinal Tracking**: Leverages EPH's 2-2-2 panel structure (households tracked over 18 months) to construct dropout trajectories. Previous work relied on point-in-time snapshots.
- **Target Population**: Focuses on ages 15+ to capture secondary-level dropouts and adult education participation, rather than theoretical school-age populations.
- **Imputation Strategy**: Explores advanced imputation techniques for income and income decile variables, beyond simple mean imputation.

---

## Repository Structure

### Architecture Standards

- **Modularity**: Logic is organized in `src/` by concern:
  - `src/data/`: Data loading, cleaning, and preprocessing
  - `src/features/`: Feature engineering functions
  - `src/models/`: Training, evaluation, and model management
  - `src/pipelines/`: ETL and training orchestration
  - `src/interpret/`: Explainability (SHAP, planned)
  - `src/api/`: FastAPI prediction endpoints (planned)
  - `src/dashboard/`: Plotly analytics dashboard

- **Code Quality**:
  - Type hints on all functions
  - Google-style docstrings for classes and methods
  - Pure preprocessing functions (input DataFrame → output DataFrame, no side effects)
  - All code and documentation in English

- **ML Best Practices**:
  - No data leakage (future information excluded)
  - Stratified K-fold cross-validation for imbalanced class handling

### Directory Layout

```
desercionAR/
├── src/
│   ├── __init__.py
│   ├── data/                    # Data loading, cleaning, preprocessing
│   ├── features/                # Feature engineering
│   ├── models/                  # Trainer, Evaluator, preprocessing factories
│   │   └── preprocessing/       # Imputer, Scaler, Encoder factories
│   ├── pipelines/               # ETL and Training orchestration
│   ├── interpret/               # Explainability logic (planned)
│   ├── api/                     # FastAPI endpoints (planned)
│   ├── dashboard/               # Plotly app
│   └── cli/                     # CLI entry points
├── configs/
│   ├── etl_pipeline.yaml        # ETL configuration
│   ├── training.yaml            # Full training configuration
│   └── training_light.yaml      # Lightweight training configuration
├── data/
│   ├── raw/                     # Original EPH data
│   ├── processed/               # ETL output (train.csv, test.csv, predict.csv)
│   └── stage/                   # Intermediate processing artifacts
├── models/                      # Model checkpoints and registry
├── notebooks/                   # Experimental notebooks (dated)
├── figures/                     # Generated visualizations
├── tests/                       # Unit and integration tests
├── pyproject.toml               # Project dependencies
├── Dockerfile
└── README.md
```

---

## ETL Pipeline

The ETL pipeline (`src/pipelines/etl.py`) processes raw EPH survey data into analysis-ready datasets for dropout prediction.

### Pipeline Stages

1. **Extract**: Downloads raw individual and household EPH microdata using `pyeph`; selects predefined features.
2. **Transform**: Filters for students aged 14+ who haven't completed secondary education; applies feature engineering transformations from `configs/etl_pipeline.yaml`.
3. **Load**: Generates target variable (dropout indicator), splits data chronologically into train/test/predict sets, prevents data leakage, applies final cleaning (binary homogenization, missing value handling). Outputs three CSV files.

### Output Datasets

- `train.csv`: Training data with target variable for model development
- `test.csv`: Test set for final model evaluation
- `predict.csv`: Unlabeled data for production predictions

### Quick Start

```bash
# Generate all datasets using the default configuration from configs/etl_pipeline.yaml
PYTHONPATH=$PWD/src poetry run python -m src.cli.etl

# Generate datasets with a custom configuration file
PYTHONPATH=$PWD/src poetry run python -m src.cli.etl --config-path configs/etl_pipeline.yaml
```

---

## Training Pipeline

The training pipeline (`src/pipelines/training.py`) orchestrates model training, evaluation, and experiment tracking via MLflow.

### Components

- **Trainer** (`trainer.py`): Builds sklearn pipelines combining preprocessing (imputation, scaling, encoding) with classifiers. Runs GridSearchCV across configurable hyperparameter grids; logs results to MLflow.

- **Evaluator** (`evaluator.py`): Computes test-set metrics (F1, accuracy, precision, recall, AUC); generates confusion matrices; logs artifacts to MLflow.

- **TrainingPipeline** (`training.py`): Orchestrates end-to-end workflow—sets up MLflow experiment, trains multiple models in parallel runs, evaluates each, and registers the best performer to the Model Registry.

### Configuration

Training is controlled via YAML:

```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "dropout_prediction"

data:
  train_path: "data/processed/train.csv"
  test_path: "data/processed/test.csv"

preprocessing:
  imputer:
    enabled: true
  scaler:
    enabled: true
  encoder:
    enabled: true

models:
  - type: "logistic_regression"
    hyperparams:
      C: [0.001, 0.01, 0.1, 1.0]
      penalty: ['l1', 'l2']
  - type: "random_forest"
    hyperparams:
      n_estimators: [8, 20, 38]

training:
  cv_folds: 5
  scoring: "f1"
```

### Quick Start

```bash
# Train with the default configuration from configs/training.yaml
PYTHONPATH=$PWD/src poetry run python -m src.cli.training

# Train with a custom configuration file
PYTHONPATH=$PWD/src poetry run python -m src.cli.training --config configs/training_light.yaml
```

### MLflow Integration

- All hyperparameters and CV scores are logged per model
- Test set metrics and confusion matrices are logged to MLflow UI
- Best model is registered to MLflow Model Registry for serving
- Access UI: `mlflow ui --backend-store-uri file:./mlruns`

---

## Setup & Dependencies

### Requirements

- Python 3.12.3
- Poetry 1.6.1+

### Installation

```bash
# Clone repository
git clone <repo-url>
cd desercionAR

# Install dependencies
poetry install

# Activate environment
eval $(poetry env activate)
```

---

## Development

### Testing

Run tests after any refactoring:

```bash
pytest tests/
```

### Adding New Models

To add a new model type:

1. Add model class to `MODEL_REGISTRY` in `trainer.py`
2. Define hyperparameters in `training.yaml` under `models`
3. Run training pipeline

### Extending Preprocessing

To add custom preprocessing:

1. Create factory function in `preprocessing/`
2. Register in `PREPROCESSOR_REGISTRY` in `trainer.py`
3. Reference in `training.yaml`

---

## Project Timeline & Status

- **v0.1.0**: Initial ETL and baseline models
- **v0.2.0**: Feature engineering and model refinement
- **Current**: MLflow integration, modular architecture, multi-model training

---

## Contact & Collaboration

For questions or contributions, please open an issue or contact the project maintainers.

---

## License

[Add license information]

