from io import StringIO

import numpy as np
import pandas as pd
from numpy.testing import assert_approx_equal

from algorithms.deliberate import user_to_graph, d_brenneis, DeliberatePredictor, Arguments

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
upeki_pretest_2_117	1			7.0	1.0	0.0	0.0			1							6.0	1.0			5	4
upeki_pretest_27170	1			6.0	1.0	0.0	0.0			0	3.0	0.0	3.0	0.0							3	5
upeki_pretest_2_90	1					1.0	0.0			1	4.0	1.0									3	0
"""), sep="\t")


def test_user2_to_graph():
    user_graph = user_to_graph(Arguments(arguments_df), users_df.iloc[1])
    assert user_graph.by_name("83").weight == 0
    assert user_graph.by_name("83").rating == 0.5
    assert user_graph.by_name("139_138").conclusion.sid == "138"
    assert user_graph.by_name("324").weight == 0.5
    assert_approx_equal(user_graph.by_name("324").rating, 0.78571428571428571429)  # 3 (of 0-6), 1 (agree)
    assert user_graph.by_name("363").weight == 0.5
    assert_approx_equal(user_graph.by_name("363").rating, 0.07142857142857142857)  # 5 (of 0-6), 0 (disagree)
    assert user_graph.by_name("361").weight == 1 / (7 + 1)
    assert user_graph.by_name("361").rating == 0
    assert user_graph.by_name("325").weight == 0
    assert user_graph.by_name("325").rating == 0.5


def test_d_brenneis():
    user_graph_1 = user_to_graph(Arguments(arguments_df), users_df.iloc[0])
    user_graph_2 = user_to_graph(Arguments(arguments_df), users_df.iloc[1])
    alpha = .4
    actual = d_brenneis({user_graph_1, user_graph_2}, alpha=alpha)
    a = 6/7/2
    b = 5/7/2
    c = 4/7/2
    d = -6/7/2
    expected = alpha * (1 - alpha) * (1/2 * a - 1/2 * c + 1/2 * b - 1/2 * d)\
               + alpha ** 2 * (1 - alpha) * (abs(8/9*1/2*.5-7/8*1/2*.5) + abs(1/9*1/2*-.5-1/8*1/2*-.5)
                                             + abs(-.5*1/2*1/2) + abs(-.5*1/2*1/2) + abs(0.5*1*1/2))
    assert_approx_equal(actual, expected)


def test_deliberate_predict_conviction():
    predictor = DeliberatePredictor()
    predictor.fit(arguments_df, users_df)
    assert predictor.predict_conviction("364", users_df.iloc[0]["username"]) == 1
    assert predictor.predict_conviction("365", users_df.iloc[0]["username"]) == 0
    assert predictor.predict_conviction("366", users_df.iloc[0]["username"]) == 0  # has no data


def test_deliberate_predict_rating():
    predictor = DeliberatePredictor()
    predictor.fit(arguments_df, users_df)
    assert_approx_equal(predictor.predict_conviction_degree("364", users_df.iloc[0]["username"]), 0.5023689554034745)
    assert_approx_equal(predictor.predict_conviction_degree("365", users_df.iloc[0]["username"]), 0)  # only one reference item
    assert predictor.predict_conviction_degree("366", users_df.iloc[0]["username"]) == 0.5  # has no data


def test_deliberate_predict_strength():
    predictor = DeliberatePredictor()
    predictor.fit(arguments_df, users_df)
    assert_approx_equal(predictor.predict_strength("364", users_df.iloc[0]["username"]), 3.502368955403474)
    assert_approx_equal(predictor.predict_strength("365", users_df.iloc[0]["username"]), 3)  # only one reference item
    assert predictor.predict_strength("366", users_df.iloc[0]["username"]) == 0  # has no data
