# Project Context: desercionAR (School Dropout Prediction)

## 1. Project Overview

- **Goal**: Predict secondary school dropout in Argentina using the EPH (Encuesta Permanente de Hogares) dataset.
- **Data Source**: EPH survey data from INDEC (Argentina), accessed via the `pyeph` library.
- **Target Variable**: `DESERTO` (Boolean), derived from `CH06` (Age), `NIVEL_ED` (Education Level), and `ESTADO` (Activity Status). `1` = Dropout, `0` = In school.

## 2. Core Principles & Workflow

- **Environment**: Use `poetry` for environment and package management.
- **ML Best Practices**:
    - **Data Leakage**: Avoid using future information or target-derived columns in features.
    - **Validation**: Use `StratifiedKFold` for cross-validation due to the imbalanced nature of the dropout class.
    - **Survey Weights**: Always use `PONDERA` for statistical analysis and as `sample_weight` in ML models to ensure national representativeness.
- **CLI Workflow**:
    - Use `!pytest` after any refactoring to ensure no regressions.
    - Use `@src` to provide context when modifying specific logic.
    - **Pipeline Execution**:
        - **ETL Pipeline**: To run the ETL pipeline, use:
            ```bash
            PYTHONPATH=$PWD/src poetry run python -m src.cli.etl --config-path configs/etl_pipeline.yaml
            ```
            The `--config-path` argument specifies the YAML configuration file for the ETL process. If omitted, it defaults to `configs/etl_pipeline.yaml`.
        - **Training Pipeline**: To run the Training pipeline, use:
            ```bash
            PYTHONPATH=$PWD/src poetry run python -m src.cli.training --config configs/training.yaml
            ```
            The `--config` argument specifies the YAML configuration file for the training process. If omitted, it defaults to `configs/training.yaml`.

-   **Lessons Learned & Watch Points**:
    1.  **CLI and Pipeline Configuration Parsing**:
        *   When integrating CLI tools (`typer`) with existing pipelines, be mindful of how configuration files are parsed and consumed by different modules.
        *   Pipelines should ideally be self-sufficient, loading their own parameters from configuration files (e.g., `etl_pipeline.yaml`'s `pipeline` section) if not explicitly provided via CLI arguments. This simplifies the CLI interface.

    2.  **`PYTHONPATH` for Python Module Execution**:
        *   When running Python modules directly (e.g., `python -m src.cli.etl`), remember to set `PYTHONPATH=$PWD/src` to ensure Python correctly resolves internal package imports within the project. This prevents `ModuleNotFoundError`s.

    3.  **Relative Data Paths in Configuration Files**:
        *   Always use project-relative paths (e.g., `data/processed/train.csv`) in configuration files (e.g., `configs/training.yaml`, `configs/etl_pipeline.yaml`) instead of absolute, user-specific paths (`~/Code/data/...`). This ensures portability across different environments and users.

## 3. Project Structure & Conventions

- **`src` Directory**: All Python source code resides in the `src/` directory.
- **Import Paths**: When importing modules, omit the `src.` prefix. The Python path is configured in `pytest.ini` and `pyproject.toml` to recognize `src` as a root.
    - **Correct**: `from pipelines.etl import ETLPipeline`
    - **Incorrect**: `from src.pipelines.etl import ETLPipeline`
- **Modularity**: Logic is organized into subdirectories within `src/`:
    - `data/`: Data loading, cleaning, and preprocessing.
    - `features/`: Feature engineering functions.
    - `models/`: Model training, evaluation, and preprocessing steps.
    - `pipelines/`: ETL and training pipelines.
    - `api/`: FastAPI application for predictions.
    - `dashboard/`: Plotly Dash application.
- **Code Style**:
    - **Type Hinting**: All new functions must include type hints.
    - **Docstrings**: Use Google-style docstrings.
    - **Language**: All code, comments, and names must be in English.

## 4. Key Components

### ETL Pipeline (`src/pipelines/etl.py`)
- **Purpose**: Processes raw EPH data into analysis-ready datasets.
- **Input**: EPH data for specified years and quarters.
- **Output**: `train.csv`, `test.csv`, and `predict.csv` in the `data/processed/` directory.
- **Key Logic**: Performs a chronological split, reserving the last quarter for the `predict` set and ensuring no data leakage between sets.

### Training Pipeline (`src/pipelines/training.py`)
- **Purpose**: Orchestrates model training, evaluation, and MLflow experiment tracking.
- **Input**: `train.csv` and `test.csv`.
- **Output**: A trained model registered in the MLflow Model Registry.
- **Key Logic**: Uses `GridSearchCV` for hyperparameter tuning and logs all results to MLflow.

### Prediction API (`src/api/main.py`)
- **Framework**: FastAPI.
- **Endpoints**:
    - `GET /health`: Health check.
    - `POST /predict`: Takes a list of records and returns dropout predictions using the latest model from the MLflow registry.

### Dashboard (`src/dashboard/`)
- **Framework**: Plotly Dash.
- **Features**: Provides data visualizations, model performance metrics, and a prediction interface.

## 5. Testing Strategy

- **Framework**: `pytest`.
- **Test Location**: Tests are located in the `tests/` directory, mirroring the `src/` structure.
- **Running Tests**: Run all tests from the project root using `poetry run pytest`.
- **Fixtures**: Use fixtures in `tests/fixtures/` to provide small, consistent datasets for testing.
- **Unit Tests**:
    - Test individual functions for data processing, feature engineering, and API endpoints.
    - Use `pytest-mock` to isolate dependencies, especially for external services like MLflow and for model loading in API tests.
- **Integration Tests**:
    - Test the full ETL and training pipelines to ensure they run end-to-end without errors.