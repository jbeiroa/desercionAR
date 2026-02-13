import yaml
import os
import pandas as pd
import mlflow
from models.trainer import Trainer
from models.evaluator import Evaluator


class TrainingPipeline:
    def __init__(self, config: dict):
        self.config = config
        mlflow.set_tracking_uri(self.config["mlflow"].get("tracking_uri"))
        mlflow.set_experiment(self.config["mlflow"].get("experiment_name"))

    def run(self):
        # Load data
        train_path = os.path.expanduser(self.config["data"]["train_path"])
        test_path = os.path.expanduser(self.config["data"]["test_path"])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Convert integer columns to float64 to handle potential NaNs and MLflow schema inference warnings
        for col in train_df.select_dtypes(include=["int64"]).columns:
            train_df[col] = train_df[col].astype("float64")
        for col in test_df.select_dtypes(include=["int64"]).columns:
            test_df[col] = test_df[col].astype("float64")

        # Drop ID columns and separate features from target
        id_cols = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE"]
        train_data = train_df.loc[:, ~train_df.columns.isin(id_cols)]
        test_data = test_df.loc[:, ~test_df.columns.isin(id_cols)]

        X_train = train_data.drop(columns="DESERTO")
        y_train = train_data["DESERTO"]
        X_test = test_data.drop(columns="DESERTO")
        y_test = test_data["DESERTO"]

        best_f1_score = -1
        best_run_id = None

        with mlflow.start_run(run_name="Training Pipeline"):
            # Train each model in a child run
            for model_config in self.config["models"]:
                with mlflow.start_run(
                    nested=True, run_name=model_config["type"]
                ) as child_run:
                    trainer = Trainer(
                        model_config=model_config,
                        preprocessor_config=self.config.get("preprocessing", {}),
                        training_config=self.config.get("training", {}),
                    )
                    results = trainer.train(X_train, y_train)

                    # Evaluate and log test metrics
                    evaluator = Evaluator(results["best_estimator"], X_test, y_test)
                    eval_results = evaluator.evaluate()
                    mlflow.log_metrics(eval_results)

                    # Check if this is the best model
                    if eval_results["f1_score"] > best_f1_score:
                        best_f1_score = eval_results["f1_score"]
                        best_run_id = child_run.info.run_id

            if best_run_id:
                print(f"Best model has F1 score: {best_f1_score:.4f}")
                print(f"Registering model from run: {best_run_id}")
                model_uri = f"runs:/{best_run_id}/model"
                mlflow.register_model(model_uri, "dropout_predictor")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full training pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to the training configuration YAML file.",
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

    pipeline = TrainingPipeline(config)
    pipeline.run()
