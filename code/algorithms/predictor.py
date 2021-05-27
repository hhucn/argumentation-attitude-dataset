from typing import List, Dict, Any

import pandas as pd


class Predictor:
    def hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        """
        returns a list of all hyperparameter combinations to be evaluated for this predictor
        """
        return [{"no_hyperparameters": None}]

    def set_hyperparameters(self, hyperparameters: Dict[str, Any]):
        for name, value in hyperparameters.items():
            self.__setattr__(name, value)

    def fit(self, arguments: pd.DataFrame, training_profiles: pd.DataFrame):
        ...

    def predict_conviction(self, statement_id: str, test_username: str) -> int:
        """
        Predict whether an argument is agreed to (1) or not (0)

        1 is returned iff the conviction degree is greater than 0.5.
        Only the username is provided as the second argument to prevent leakage of the test set into the prediction
        """
        return round(self.predict_conviction_degree(statement_id, test_username))

    def predict_conviction_degree(self, statement_id: str, test_username: str) -> float:
        """
        Predict the degree to which an argument is agreed to (1) or not (0), via a value in [0, 1]

        Only the username is provided as the second argument to prevent leakage of the test set into the prediction
        """
        ...

    def predict_strength(self, statement_id: str, test_username: str) -> float:
        """
        Predict how strong an argument is considered (values in [0,6])

        Only the username is provided as the second argument to prevent leakage of the test set into the prediction
        """
        ...
