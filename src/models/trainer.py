import numpy as np
import pandas as pd
import mlflow
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
    def __init__(self, model_config: dict, preprocessor_config: dict, training_config: dict):
        self.model_config = model_config
        self.preprocessor_config = preprocessor_config
        self.training_config = training_config
        self.pipeline = self._make_pipeline()
        self.param_grid = self._build_param_grid()
        self.best_model = None
        self.grid_search = None

    def _make_pipeline(self) -> Pipeline:
        """
        Builds a sequential scikit-learn pipeline from preprocessors
        defined in the config file.
        """
        steps = []
        for name, maker_key in self.preprocessor_config.items():
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
            
        model_class = MODEL_REGISTRY[model_type]
        return model_class(**base_params)

    def _parse_hyperparam_values(self, param_spec) -> list[object]:
        """Convert YAML hyperparam specs to lists."""
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

    def _build_param_grid(self) -> dict:
        """Builds a parameter grid for a single model config."""
        param_dict = {}
        model_type = self.model_config['type']
        base_params = self.model_config.get('base_params', {}).copy()
        hyperparams_spec = self.model_config.get('hyperparams', {}).copy()

        if 'estimator' in base_params:
            nested_model_config = base_params.pop('estimator')
            base_params['estimator'] = self._instantiate_model(
                nested_model_config['type'],
                nested_model_config.get('base_params', {})
            )

        model_instance = self._instantiate_model(model_type, base_params)
        param_dict['classifier'] = [model_instance]

        for param_name, param_spec in hyperparams_spec.items():
            param_values = self._parse_hyperparam_values(param_spec)
            full_param_name = f"classifier__{param_name}"
            param_dict[full_param_name] = param_values

        return param_dict

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Run GridSearchCV with the model's param_grid and return results.
        """
        mlflow.sklearn.autolog(
            log_input_examples=False,
            log_model_signatures=False
        )

        cv_folds = self.training_config.get('cv_folds', 3)
        scoring = self.training_config.get('scoring', 'f1_macro')
        mlflow.log_params({'cv_folds': cv_folds, 'scoring': scoring})

        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

        print(f"Starting GridSearchCV for {self.model_config['type']}...")
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        print("GridSearchCV finished.")

        return {
            'best_estimator': self.best_model,
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
        }