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