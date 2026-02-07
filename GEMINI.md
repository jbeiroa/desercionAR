# Project Context: desercionAR (School Dropout Prediction)

## 0. Environment management
Use `poetry` for environment and package management.

## 1. Project Goal
Predicting secondary school dropout in Argentina using the EPH (Encuesta Permanente de Hogares) dataset. The project involves data acquisition via `pyeph`, preprocessing for social indicators, and training predictive models.

## 2. Domain Knowledge (EPH & pyeph)
- **Data Source:** EPH is a household survey from INDEC (Argentina).
- **Core Library:** `pyeph`. Key usage: `pyeph.get(data="eph", year=YYYY, period=Q, base_type='individual')`.
- **Target Variable (Dropout):** Derived from `CH06` (Age), `NIVEL_ED` (Education Level), and `ESTADO` (Activity Status). 
- **Full Variable Map:** See `@Variables.md` for definitions of all EPH codes (e.g., IV1, CH06).
- **Core Groups:**
    - **Demographics:** Age (CH06), Sex (CH04), and household composition are primary predictors.
    - **Socioeconomics:** Use NBI flags (NBI_*) and Income Deciles (DECCFR) as proxies for SES.
    - **Target Variable:** `DESERTO` (Boolean). 1 = Dropout, 0 = In school.
- **Handling Survey Weights:** Always use `PONDERA` for statistical analysis and `sample_weight` in ML models to ensure Argentine national representativeness.

## 3. Refactoring Standards
- **Modularity:** Move and restructure logic into `src/` directory:
  - `src/data/`: `pyeph` wrappers and loading logic.
  - `src/features/`: Feature engineering and survey-specific transformations.
  - `src/models/`: Trainer and Evaluator classes (scikit-learn/XGBoost/CatBoost).
  - `src/pipelines/`: ETL and training pipelines.
  - `src/interpret`: SHAP and/or explanibility logic (not implemented yet)
  - `src/api/`: FastAPI for prediction (dummy implementation).
  - `src/dashboard/`: plotly app.
- **Type Hinting:** All new Python functions must include type hints.
- **Documentation:** Use Google-style docstrings for all classes and functions.
- **Purity:** Keep preprocessing functions "pure" (input DataFrame -> output DataFrame) to avoid side effects.
- **Comments**: Add code comments sparingly. Focus on *why* something is done, especially for complex logic, rather than *what* is done. Only add high-value comments if necessary for clarity. *NEVER* talk to the user or describe your changes through comments.
- **Language:** All code, including comments, variable and function names, must be in english.

## 4. ML Best Practices
- **Data Leakage:** Never use information from the future or target-derived columns in features.
- **Validation:** Use `StratifiedKFold` because dropout is likely an imbalanced class.

## 5. CLI Workflow
- Use `/checkpoint` before applying structural changes to the `src/` directory.
- Use `!pytest` after refactoring modules to ensure no regression.
- Use `@src` to provide context when modifying specific logic.

## 6. Directory structure
```
desercionAR/
├── src/
│   ├── __init__.py
│   ├── data/			      # ETL: loading, cleaning, preprocessing
│   ├── features/		    # Feature engineering functions
│   ├── models/			    # Training, evaluation
│   ├── interpret/		  # SHAP or explainability logic
│   ├── api/			      # FastAPI endpoints for prediction
│   ├── pipelines/			# ETL and training pipelines
│   └── dashboard/		  # plotly app
├── tests/				      # Unit & integration tests
├── notebooks/			    # Experimental notebooks (dated)
├── data/				        # gitignored, upload train-test-val dataset to hf
│   ├── raw/
│   ├── preprocessed/
│   └── stage/
├── artifacts/			    # Logs, transformers, metrics, etc.
├── figures/
├── docs/
├── pyproject.toml
├── Dockerfile
├── README.md
└── .github/
	   └── workflows/	# CI/CD pipelines
```
  
## 7. ETL Pipeline

The ETL pipeline, located in `src/pipelines/etl.py`, processes raw EPH (Encuesta Permanente de Hogares) data to create analysis-ready datasets for predicting student dropout. It is designed to be executed via the command-line interface.

### Workings

The pipeline operates in three main stages:

1.  **Extract**: It downloads raw individual and household EPH data for specified years and quarters using the `pyeph` library. It then selects a predefined set of features from these raw files.

2.  **Transform**:
    *   The data is filtered to focus on students aged 14 and over who have not yet completed secondary education.
    *   It applies a series of feature engineering steps defined in `configs/etl_pipeline.yaml`, such as creating new variables based on the household head and their spouse.

3.  **Load**: This stage handles target creation, data splitting, and final processing.
    *   **Target Generation**: The `DESERTO` target variable is created by comparing a student's enrollment status across two consecutive quarters. A student is marked as a dropout (`DESERTO` = 1) if they were enrolled (`CH10` = 1) in one quarter and not enrolled (`CH10` = 2) in the next. This is performed only on the training/testing set.
    *   **Data Splitting**: The pipeline performs a strict chronological split. The most recent quarter of data is reserved as the `predict.csv` set. All prior quarters are used to generate the `train.csv` and `test.csv` sets.
    *   **Leakage Prevention**: To correctly handle the panel data, any individual who appears in the `predict` set is explicitly removed from the `train` and `test` sets.
    *   **Post-processing**: It applies final cleaning steps, including:
        *   Homogenizing binary columns (e.g., converting 'SI'/'NO' to 1/0).
        *   Calculating the geographical distance of each region to the capital.
        *   Handling missing values and dropping intermediate or unnecessary columns.
    *   **Output**: The final datasets (`train.csv`, `test.csv`, `predict.csv`) are saved to the `data/processed/` directory.

### Example Usage

The pipeline is executed from the project root directory using the CLI script.

```bash
# Run the ETL with default settings to generate all three datasets
python -m src.cli.etl

# Run a custom ETL for 2023 data to generate only a prediction set
python -m src.cli.etl --years 2023 --quarters 1 --quarters 2 --dataset-type predict
```

# 8. Training Pipeline

The training pipeline (`src/pipelines/training.py`) orchestrates model training, evaluation, and experiment tracking via MLflow.

## Components

- **Trainer** (`trainer.py`): Builds sklearn pipelines combining preprocessing (imputation, scaling, encoding) with classifiers. Runs GridSearchCV across configurable hyperparameter grids; logs results to MLflow.

- **Evaluator** (`evaluator.py`): Computes test-set metrics (F1, accuracy, precision, recall, AUC); generates confusion matrices; logs artifacts to MLflow.

- **TrainingPipeline** (`training.py`): Orchestrates end-to-end workflow—sets up MLflow experiment, trains multiple models in parallel runs, evaluates each, and registers the best performer to the Model Registry.

## Example config file

# File: configs/training_light.yaml

```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "Light Training desercionAR"

data:
  train_path: "~/Code/data/desercionAR/train.csv"
  test_path: "~/Code/data/desercionAR/test.csv"

# Preprocessor step names and the registry keys for their functions.
# The pipeline will be built sequentially in this order.
preprocessors:
  imputer: "make_imputer"
  scaler: "make_scaler"
  encoder: "make_encoder"

# A single model for a quick test run
models:
  - type: "logistic_regression"
    base_params:
      random_state: 42
      solver: "liblinear"
    hyperparams:
      C: [0.1, 1.0, 10.0]
      penalty: ["l1", "l2"]

# Configuration for the GridSearchCV process
training:
  cv_folds: 3
  scoring: "f1_macro"

### Usage example
```bash
# Train with default configuration
python -m src.cli.training

# Train with custom configuration
python -m src.cli.training --config configs/training_light.yaml
```

# 9. Dashboard

Dashboard app uses `Plotly-Dash` with three pages:
1. Home page showing a presentation message describing the project and a summary of the data.
  - A map of Argentina divided by color-scaled regions with the number of dropouts. Selecting a region should filter plots to show data from that region only.
  - Plots of some important features for the model.
  - Data on model performance.
2. An Analytics page that allow users to see data distribution for different variables of the dataset.
3. A page that uses an entry form with some predefined values to get a singular prediction using the api.

# 10. Development, Testing, and Deployment Plan

This section outlines a comprehensive strategy for creating a robust test suite, implementing CI/CD pipelines, and deploying the application to AWS with a focus on cost-effectiveness.

## 10.1. Testing Strategy (Pytest)

A full test suite ensures code quality, prevents regressions, and validates the logic of data processing and modeling. We will use the `pytest` framework.

1.  **Setup Test Environment:**
    *   Create a `tests/` directory if it doesn't already exist.
    *   Populate `tests/` with subdirectories mirroring the `src/` structure (e.g., `tests/data`, `tests/pipelines`, `tests/api`).
    *   Create a `tests/fixtures/` directory to store sample data for tests. This data should be a small, representative subset of the EPH data, containing known values to assert against.

2.  **Unit Tests:**
    *   **`src/data`:** Write tests for data cleaning and preprocessing functions. Use fixture data to test edge cases, such as missing values or unexpected data types.
    *   **`src/features`:** Test each feature engineering function in isolation. Ensure the output DataFrame has the correct schema and values based on a known input.
    *   **`src/models/preprocessing`:** Test imputer, scaler, and encoder transformers to ensure they behave as expected.
    *   **`src/api`:** Write unit tests for the FastAPI endpoints. Use `pytest-mock` to mock the model loading and prediction functions, testing only the API logic (request/response handling, status codes).

3.  **Integration Tests:**
    *   **`src/pipelines/etl.py`:** Write an integration test for the ETL pipeline. This test will run the pipeline on a small, controlled dataset (from `tests/fixtures/`) and assert that the output files (`train.csv`, `test.csv`, `predict.csv`) are created correctly and have the expected content.
    *   **`src/pipelines/training.py`:** Write an integration test for the training pipeline. This will run the training process on the test fixture data, ensuring the pipeline executes end-to-end without errors and produces a model artifact.

## 10.2. CI/CD Strategy (GitHub Actions)

We will use GitHub Actions to automate testing and prepare for deployment. A workflow file will be created under `.github/workflows/ci.yml`.

1.  **Continuous Integration (CI) Workflow:**
    *   **Trigger:** The workflow will trigger on every `push` to the main branch and on every `pull_request`.
    *   **Jobs:**
        1.  **Lint & Type Check:**
            *   Set up a Python environment and install dependencies using `poetry install`.
            *   Run a linter like `ruff` or `flake8` to enforce code style.
            *   Run `mypy` to perform static type checking.
        2.  **Test:**
            *   Set up a Python environment and install dependencies.
            *   Run the entire `pytest` suite.
            *   (Optional) Collect and upload test coverage reports.

2.  **Continuous Deployment (CD) Workflow:**
    *   **Trigger:** The workflow will trigger on a `push` to a specific branch (e.g., `release`) or manually via `workflow_dispatch`.
    *   **Jobs:**
        1.  **Build:**
            *   Build a Docker container for the FastAPI application.
            *   Build a Docker container for the Plotly Dash dashboard.
        2.  **Push:**
            *   Push the Docker images to a container registry (e.g., Amazon ECR).
        3.  **Deploy:**
            *   Use the AWS CLI to deploy the new container images to their respective services (e.g., AWS App Runner).

## 10.3. AWS Deployment (Cost-Effective)

The goal is a low-cost, scalable deployment. We will primarily use serverless and managed services.

1.  **Data and Artifact Storage (Amazon S3):**
    *   Create an S3 bucket to store:
        *   Raw and processed data.
        *   Trained model artifacts (`.pkl` files).
        *   MLflow experiment logs (can be configured to use S3 as a backend).
    *   Update the ETL and Training pipelines to read from and write to this S3 bucket instead of the local filesystem.

2.  **API Deployment (AWS App Runner):**
    *   Package the FastAPI application in `src/api` into a Docker container.
    *   Create an AWS App Runner service linked to the ECR repository where the API image is stored.
    *   App Runner will automatically handle scaling (including scaling to zero to save costs when not in use), load balancing, and HTTPS.

3.  **Dashboard Deployment (AWS App Runner):**
    *   Package the Plotly Dash application in `src/dashboard` into a Docker container.
    *   Deploy it as a separate AWS App Runner service, similar to the API.

4.  **Model Training (Manual/Scheduled):**
    *   For maximum cost savings, the training pipeline can be run locally.
    *   For an automated approach, a GitHub Actions workflow can be created that can be manually triggered to run the training pipeline. This workflow would run on a GitHub-hosted runner, execute the training script (which pulls data from S3), and upload the resulting model artifact back to S3.

5.  **Security and Configuration:**
    *   Use AWS Secrets Manager or Parameter Store to manage sensitive information like database credentials or API keys (if any).
    *   Use IAM roles and policies to grant granular permissions to services (e.g., allowing the App Runner service to access the S3 bucket).