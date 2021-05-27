import pandas as pd

from algorithms.predictor import Predictor


class MajorityPredictor(Predictor):
    """
    Predictor which always predicts the majority class, or for regression the average
    """

    def __init__(self):
        self.mean_convictions = {}
        self.mean_strengths = {}

    def fit(self, arguments: pd.DataFrame, training_profiles: pd.DataFrame):
        self.mean_convictions = {}
        self.mean_strengths = {}
        for column_name in training_profiles.columns:
            if "statement_attitude_" in column_name:
                self.mean_convictions[column_name.split("_")[-1]] = training_profiles[column_name].dropna().mean()
            if "argument_rating_" in column_name:
                self.mean_strengths[column_name.split("_")[-1]] = training_profiles[column_name].dropna().mean()

    def predict_conviction_degree(self, statement_id: str, test_username: str) -> float:
        return self.mean_convictions[statement_id]

    def predict_strength(self, statement_id: str, test_username: str) -> float:
        return self.mean_strengths[statement_id]
