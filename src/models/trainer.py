import os
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from .preprocessing.encoder import make_encoder
from .preprocessing.imputer import make_imputer
from .preprocessing.scaler import make_scaler


PREPROCESSOR_REGISTRY = {
    "make_imputer": make_imputer,
    "make_scaler": make_scaler,
    "make_encoder": make_encoder,
}

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "bagging": BaggingClassifier,
}

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.pipeline = self._make_pipeline()
        self.param_grid = self._build_param_grid()
        self.best_model = None
        self.grid_search = None

    def _make_pipeline(self) -> Pipeline:
        """
        Builds a sequential scikit-learn pipeline from preprocessors
        defined in the config file. Assumes each preprocessor is a
        self-contained ColumnTransformer.
        """
        steps = []
        preprocessors = self.config.get('preprocessors', {})
        for name, maker_key in preprocessors.items():
            if maker_key not in PREPROCESSOR_REGISTRY:
                raise ValueError(f"Preprocessor function '{maker_key}' not found in registry.")
            
            maker_func = PREPROCESSOR_REGISTRY[maker_key]
            steps.append((name, maker_func()))

        # Add a placeholder classifier; this will be replaced by GridSearchCV
        steps.append(('classifier', LogisticRegression()))
        
        return Pipeline(steps=steps)

    def _instantiate_model(self, model_type: str, base_params: dict) -> object:
        """Instantiates a model class with its base parameters."""
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        
        # If a parameter for the base model is another model, instantiate it first
        if 'estimator' in base_params:
            nested_model_config = base_params.pop('estimator')
            base_params['estimator'] = self._instantiate_model(
                nested_model_config['type'],
                nested_model_config.get('base_params', {})
            )
            
        model_class = MODEL_REGISTRY[model_type]
        return model_class(**base_params)

    def _parse_hyperparam_values(self, param_spec) -> list[object]:
        """Convert YAML hyperparam specs (range, linspace, logspace) to lists."""
        if isinstance(param_spec, list):
            return list(param_spec)
        elif isinstance(param_spec, dict):
            spec_type = param_spec.get('type')
            if spec_type == 'range':
                return list(range(param_spec['start'], param_spec['stop'], param_spec['step']))
            elif spec_type == 'linspace':
                return np.linspace(param_spec['start'], param_spec['stop'], param_spec['num']).tolist()
            elif spec_type == 'logspace':
                return np.logspace(param_spec['start'], param_spec['stop'], param_spec['num']).tolist()
            else:
                raise ValueError(f"Unknown hyperparam type: {spec_type}")
        else:
            return [param_spec]

    def _build_single_model_grid(self, model_config: dict, prefix: str) -> dict:
        """
        Recursively builds a parameter dictionary for a single model config,
        handling nested estimators.
        """
        param_dict = {}
        model_type = model_config['type']
        base_params = model_config.get('base_params', {}).copy()
        hyperparams_spec = model_config.get('hyperparams', {}).copy()

        # If a parameter for the base model is another model, instantiate it first
        if 'estimator' in base_params:
            nested_model_config = base_params.pop('estimator')
            base_params['estimator'] = self._instantiate_model(
                nested_model_config['type'],
                nested_model_config.get('base_params', {})
            )

        # Instantiate the current level's model
        model_instance = self._instantiate_model(model_type, base_params)
        param_dict[prefix] = [model_instance]

        # Parse hyperparameters for the current level
        for param_name, param_spec in hyperparams_spec.items():
            param_values = self._parse_hyperparam_values(param_spec)
            full_param_name = f"{prefix}__{param_name}"
            param_dict[full_param_name] = param_values

        return param_dict

    def _build_param_grid(self) -> list[dict]:
        """Converts YAML model configs to a GridSearchCV-compatible param_grid list."""
        param_grid = []
        for model_config in self.config.get('models', []):
            model_param_dict = self._build_single_model_grid(model_config, 'classifier')
            param_grid.append(model_param_dict)
        return param_grid

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Run GridSearchCV with param_grid and return results.
        """
        training_config = self.config.get('training', {})
        cv_folds = training_config.get('cv_folds', 3)
        scoring = training_config.get('scoring', 'f1_macro')

        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

        print("Starting GridSearchCV...")
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        print("GridSearchCV finished.")

        return {
            'best_estimator': self.best_model,
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_light.yaml",
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    print(f"Loading configuration from: {args.config}")
    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading YAML config: {e}")
            exit(1)

    # Load and prepare data
    data_path = os.path.expanduser(config['train_data_path'])
    print(f"Loading data from: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Training data not found at {data_path}")
        exit(1)

    # Drop ID columns and separate features from target
    # Note: This assumes 'DESERTO' is the target. A more robust solution
    # might define the target and ID columns in the config file.
    id_cols = ['CODUSU', 'NRO_HOGAR', 'COMPONENTE', 'ANO4', 'TRIMESTRE']
    train_data = data.loc[:, ~data.columns.isin(id_cols)]
    
    X_train = train_data.drop(columns='DESERTO')
    y_train = train_data['DESERTO']

    # Instantiate and run the trainer
    print("Instantiating trainer...")
    trainer = Trainer(config=config)

    results = trainer.train(X_train, y_train)

    # Print the results
    print("\n--- Training Results ---")
    print(f"Best Score ({config['training']['scoring']}): {results['best_score']:.4f}")
    print("Best Parameters:")
    # Pretty print the best parameters
    best_params = results['best_params']
    for param, value in best_params.items():
        # Shorten the classifier representation for readability
        if isinstance(value, (LogisticRegression, DecisionTreeClassifier, BaggingClassifier)):
            value_str = value.__class__.__name__
        else:
            value_str = value
        print(f"  {param}: {value_str}")


