from io import StringIO

import pandas as pd
from numpy.testing import assert_approx_equal

from algorithms.majority import MajorityPredictor
from main import evaluate_conviction, evaluate_strength, evaluate_precision_n

arguments_df = pd.read_csv(StringIO("""statement_id	conclusions
138	83(+)
139	138(+);84(+)
83	
84	83(+)
85	83(+);96(+)
86	83(+)
87	83(+);101(-);95(-)
88	83(+)
324	
325	324(+)
326	324(+)
361	324(-)
362	324(-)
363	
364	363(+)
365	363(+)
367	363(+)
368	363(+)
"""), sep="\t")

users_df = pd.read_csv(StringIO("""username	statement_attitude_324	argument_rating_325	statement_attitude_325	argument_rating_326	statement_attitude_326	argument_rating_361	statement_attitude_361	argument_rating_362	statement_attitude_362	statement_attitude_363	argument_rating_364	statement_attitude_364	argument_rating_365	statement_attitude_365	argument_rating_366	statement_attitude_366	argument_rating_367	statement_attitude_367	argument_rating_368	statement_attitude_368	position_rating_after_324	position_rating_after_363
upeki_pretest_2_117	1	0	0	7.0	1.0	0.0	0.0	0		1							6.0	1.0			5	4
upeki_pretest_27170	1	0	0	6.0	1.0	0.0	0.0		0	0	3.0	0.0	3.0	0.0	0	0	0	0	0		3	5
upeki_pretest_2_91	1	0	0	0	0	1.0	0.0			1	4.0	1.0									3	0
upeki_pretest_2_92	1	0	0	0	0	1.0	0.0			1	3.0	1.0								0	2	1
upeki_pretest_2_93	1	0	0	0	0	1.0	0.0			1	2.0	1.0									3	2
upeki_pretest_2_94	1	0	0	0	0	1.0	0.0			1	1.0	1.0									1	3
"""), sep="\t")


def test_evaluate_conviction():
    predictor = MajorityPredictor()
    predictor.fit(arguments_df, users_df)
    result = evaluate_conviction(predictor, users_df, 0, 1000)
    assert_approx_equal(result, 0.89444444)


def test_evaluate_conviction_with_range_all_correct():
    predictor = MajorityPredictor()
    predictor.fit(arguments_df, users_df)
    result = evaluate_conviction(predictor, users_df, 324, 324)
    assert result == 1


def test_evaluate_strength():
    predictor = MajorityPredictor()
    predictor.fit(arguments_df, users_df)
    result = evaluate_strength(predictor, users_df, 0, 1000)
    assert_approx_equal(result, 1.5353917)


def test_evaluate_strength_with_range_all_correct():
    predictor = MajorityPredictor()
    predictor.fit(arguments_df, users_df)
    result = evaluate_strength(predictor, users_df, 325, 325)
    assert_approx_equal(result, 0)


def test_evaluate_precision_n():
    predictor = MajorityPredictor()
    predictor.fit(arguments_df, users_df)
    result = evaluate_precision_n(predictor, users_df, 3, 0, 1000)
    assert_approx_equal(result, 0.8888888888888888)


def test_evaluate_precision_n_with_range_all_correct():
    predictor = MajorityPredictor()
    predictor.fit(arguments_df, users_df)
    result = evaluate_precision_n(predictor, users_df, 1, 324, 324)
    assert_approx_equal(result, 1)
