import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    def __init__(self, model: object, X_test: pd.DataFrame, y_test: np.ndarray):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self) -> dict:
        """
        Evaluates the model on the test set and returns performance metrics.
        """
        y_pred = self.model.predict(self.X_test)

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average="macro"),
            "recall": recall_score(self.y_test, y_pred, average="macro"),
            "f1_score": f1_score(self.y_test, y_pred, average="macro"),
        }

        return metrics
