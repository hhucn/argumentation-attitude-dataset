import zlib

import pandas as pd

from algorithms.predictor import Predictor


class RandomPredictor(Predictor):
    """
    Predictor which always predicts a random value
    """

    def __init__(self):
        self.mean_convictions = {}
        self.mean_strengths = {}

    def fit(self, arguments: pd.DataFrame, training_profiles: pd.DataFrame):
        return

    def predict_conviction_degree(self, statement_id: str, test_username: str) -> float:
        return (zlib.adler32(f"{statement_id}{test_username}".encode("UTF8")) % 100) / 100

    def predict_strength(self, statement_id: str, test_username: str) -> float:
        return (zlib.adler32(f"{statement_id}{test_username}".encode("UTF8")) % 70) / 10
