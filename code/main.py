"""
Task: based on the data from a previous point of time, predict which arguments are considered convincing or not
"""
import sys

import numpy as np
import pandas as pd

from algorithms.deliberate import DeliberatePredictor
from algorithms.majority import MajorityPredictor
from algorithms.predictor import Predictor
from algorithms.random import RandomPredictor
from common import ProgressBar


def evaluate_conviction(predictor: Predictor, test_profiles: pd.DataFrame, target_id_start: int, target_id_end: int) -> float:
    """
    calculates the macro accuracy for predicting argument conviction (0 or 1)

    only arguments with ground truths and ids falling in the given ranges are evaluated on
    """
    accuracies = []
    for i, test_profile in ProgressBar("conviction", max=len(test_profiles)).iter(test_profiles.iterrows()):
        evaluation_statements = [(k.split("_")[-1], v) for k, v in test_profile.dropna().iteritems()
                                 if "statement_attitude_" in k and target_id_start <= int(k.split("_")[-1]) <= target_id_end]
        assert len(evaluation_statements) >= 1, f"not enough evaluation statements for user {test_profile['username']}, is your argument id range and your dataset okay?"
        correct_prediction = 0
        for statement_id, ground_truth in evaluation_statements:
            prediction = predictor.predict_conviction(statement_id, test_profile["username"])
            if prediction == ground_truth:
                correct_prediction += 1
        accuracy = correct_prediction / len(evaluation_statements)
        accuracies.append(accuracy)
    return float(np.mean(accuracies))


def evaluate_strength(predictor: Predictor, test_profiles: pd.DataFrame, target_id_start: int, target_id_end: int) -> float:
    """
    calculates the average mean squared error for predicting an argument's strength (value in [0, 6])

    only arguments with ground truths and ids falling in the given ranges are evaluated on
    """
    rmses = []
    for i, test_profile in ProgressBar("strength", max=len(test_profiles)).iter(test_profiles.iterrows()):
        evaluation_statements = [(k.split("_")[-1], v) for k, v in test_profile.dropna().iteritems()
                                 if "argument_rating_" in k and target_id_start <= int(k.split("_")[-1]) <= target_id_end]
        assert len(evaluation_statements) >= 1, f"not enough evaluation statements for user {test_profile['username']}, is your argument id range and your dataset okay?"
        squared_errors = 0
        for statement_id, ground_truth in evaluation_statements:
            prediction = predictor.predict_strength(statement_id, test_profile["username"])
            squared_errors += (prediction - ground_truth) ** 2
        rmse = np.sqrt(squared_errors / len(evaluation_statements))
        rmses.append(rmse)
    return float(np.mean(rmses))


def evaluate_precision_n(predictor: Predictor, test_profiles: pd.DataFrame, n: int, target_id_start: int,
                         target_id_end: int) -> float:
    """
    Calculates the averaged precision@n for argument conviction

    How many of the top-n recommended items (only considering items for which we have a ground truth) are considered convincing by the user?
    Note that an algorithm may recommend less than n items.
    Only arguments falling in the given id range are evaluated on
    """
    assert n >= 1
    precisions = []
    for i, test_profile in ProgressBar(f"p@{n}", max=len(test_profiles)).iter(test_profiles.iterrows()):
        evaluation_statements = [(k.split("_")[-1], v) for k, v in test_profile.dropna().iteritems()
                                 if "statement_attitude_" in k and target_id_start <= int(k.split("_")[-1]) <= target_id_end]
        assert len(evaluation_statements) >= n, f"not enough evaluation statements for user {test_profile['username']}, is your argument id range and your dataset okay?"
        predicted_relevant = []
        for statement_id, ground_truth in evaluation_statements:
            prediction = predictor.predict_conviction_degree(statement_id, test_profile["username"])
            if prediction > .5:
                predicted_relevant.append((prediction, ground_truth, statement_id))  # statement_id as tie breaker for reproducibility
        correct_predictions = 0
        predicted_relevant_top_n = sorted(predicted_relevant)[-n:]
        for prediction, ground_truth, _ in predicted_relevant_top_n:
            if ground_truth == 1:
                correct_predictions += 1
        if len(predicted_relevant_top_n) > 0:
            precision = correct_predictions / len(predicted_relevant_top_n)
            precisions.append(precision)
    return float(np.mean(precisions))


def main(task: str, algorithm: str, arguments_filename: str, training_profiles_filename: str,
         test_profiles_filename: str, target_id_start: int, target_id_end: int):
    arguments = pd.read_csv(arguments_filename)
    training_profiles = pd.read_csv(training_profiles_filename)
    test_profiles = pd.read_csv(test_profiles_filename)

    if algorithm == "majority":
        predictor = MajorityPredictor()
    elif algorithm == "random":
        predictor = RandomPredictor()
    elif algorithm == "deliberate":
        predictor = DeliberatePredictor()
    else:
        raise ValueError(f"algorithm {algorithm} is unknown")

    predictor.fit(arguments, training_profiles)

    values = []
    for hyperparameters in predictor.hyperparameter_combinations():
        predictor.set_hyperparameters(hyperparameters)
        print(predictor.__class__.__name__, hyperparameters)

        if task == "conviction":
            macro_accuracy = evaluate_conviction(predictor, test_profiles, target_id_start, target_id_end)
            print(macro_accuracy)
            values.append((macro_accuracy, hyperparameters))
        elif task == "strength":
            armse = evaluate_strength(predictor, test_profiles, target_id_start, target_id_end)
            print(armse)
            values.append((armse, hyperparameters))
        elif task == "precision@3":
            precision = evaluate_precision_n(predictor, test_profiles, 3, target_id_start, target_id_end)
            print(precision)
            values.append((precision, hyperparameters))
        else:
            raise ValueError(f"task {task} is unknown")

    if task in {"strength"}:
        min_value = min(v for v, _ in values)
        print(f"min value {min_value} for", [p for v, p in values if v == min_value])
    elif task in {"conviction", "precision@3"}:
        max_value = max(v for v, _ in values)
        print(f"max value {max_value} for", [p for v, p in values if v == max_value])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]))
