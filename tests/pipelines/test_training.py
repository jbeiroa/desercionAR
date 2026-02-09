import pytest
import yaml
import pandas as pd
from unittest.mock import patch, MagicMock
from pipelines.training import TrainingPipeline

@pytest.fixture
def temp_data(tmp_path):
    """Create dummy train and test CSV files in a temporary directory."""
    train_dir = tmp_path / "data"
    train_dir.mkdir()
    train_path = train_dir / "train.csv"
    test_path = train_dir / "test.csv"

    # Create dummy data with required columns
    train_df = pd.DataFrame({
        'CODUSU': ['A', 'B', 'C'],
        'NRO_HOGAR': [1, 1, 2],
        'COMPONENTE': [1, 2, 1],
        'ANO4': [2023, 2023, 2023],
        'TRIMESTRE': [1, 1, 1],
        'DESERTO': [0, 1, 0],
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [0.4, 0.5, 0.6]
    })
    test_df = pd.DataFrame({
        'CODUSU': ['D', 'E'],
        'NRO_HOGAR': [3, 4],
        'COMPONENTE': [1, 1],
        'ANO4': [2023, 2023],
        'TRIMESTRE': [2, 2],
        'DESERTO': [1, 0],
        'feature1': [0.7, 0.8],
        'feature2': [0.9, 1.0]
    })

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_path, test_path

@pytest.fixture
def training_config(tmp_path, temp_data):
    """Create a temporary training_light.yaml config file."""
    train_path, test_path = temp_data
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "training_light.yaml"

    config = {
        'mlflow': {
            'tracking_uri': f"file:{tmp_path}/mlruns",
            'experiment_name': 'test_experiment'
        },
        'data': {
            'train_path': str(train_path),
            'test_path': str(test_path)
        },
        'models': [
            {
                'type': 'logistic_regression',
                'base_params': {
                    'random_state': 42,
                    'solver': 'liblinear'
                },
                'hyperparams': {
                    'C': [0.1, 1.0],
                    'penalty': ['l1', 'l2']
                }
            }
        ],
        'training': {
            'cv_folds': 2,
            'scoring': 'f1_macro'
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    return config_path

@patch('pipelines.training.Trainer')
@patch('pipelines.training.mlflow')
def test_training_pipeline_run(mock_mlflow, mock_trainer, training_config, temp_data):
    """
    Test the TrainingPipeline run method.
    
    This test ensures that the pipeline initializes correctly, runs without errors,
    and that MLflow tracking is called as expected.
    """
    # Mock MLflow experiment setup
    mock_mlflow.get_experiment_by_name.return_value = None
    mock_mlflow.create_experiment.return_value = "test_experiment_id"
    
    # Mock the context manager for runs
    main_run_mock = MagicMock()
    main_run_mock.info.run_id = "main_run_id"
    child_run_mock = MagicMock()
    child_run_mock.info.run_id = "child_run_id"
    
    # Simulate entering the 'with' statements
    mock_mlflow.start_run.side_effect = [
        MagicMock(__enter__=MagicMock(return_value=main_run_mock)),
        MagicMock(__enter__=MagicMock(return_value=child_run_mock))
    ]

    # Mock the Trainer class
    mock_trainer_instance = mock_trainer.return_value
    mock_trainer_instance.train.return_value = {
        'best_estimator': MagicMock(),
        'best_params': {},
        'best_score': 0.9
    }
    
    # Mock the Evaluator
    with patch('pipelines.training.Evaluator') as mock_evaluator:
        mock_evaluator_instance = mock_evaluator.return_value
        mock_evaluator_instance.evaluate.return_value = {
            'f1_score': 0.95,
            'accuracy': 0.92
        }

        # Load config from the temporary file
        with open(training_config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Initialize and run the pipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()

        # Assertions
        mock_mlflow.set_tracking_uri.assert_called_once_with(config['mlflow']['tracking_uri'])
        mock_mlflow.set_experiment.assert_called_once_with(config['mlflow']['experiment_name'])
        
        # Check that runs were started
        assert mock_mlflow.start_run.call_count == 2
        
        # Check that the trainer was initialized and run
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        
        # Check that the evaluator was used
        mock_evaluator.assert_called_once()
        mock_evaluator_instance.evaluate.assert_called_once()
        
        # Check that metrics were logged
        mock_mlflow.log_metrics.assert_called_once_with({'f1_score': 0.95, 'accuracy': 0.92})
        
        # Check that the best model was registered
        mock_mlflow.register_model.assert_called_once_with(
            "runs:/child_run_id/model", "dropout_predictor"
        )

