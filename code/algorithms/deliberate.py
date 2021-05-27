from typing import Optional, List, Set, FrozenSet, Tuple, Dict, Union, Any

import numpy as np
import pandas as pd
from edn_format import ImmutableDict
from memoization import cached

from algorithms.predictor import Predictor
from common import ProgressBar
from config import CONFIG

ttl = 60 * 60  # 1 hour


class Statement:
    """Represents a weighted argumentation graph, based on the definition by Brenneis et al. (2020)"""

    def __init__(self, premises=None, conclusion: Optional["Statement"] = None,
                 rating: float = 0.5, weight: float = 0, sid: str = "", username=None):
        if premises is None:
            premises = []
        self.premises: Optional[List["Statement"]] = premises
        self.conclusion: Optional["Statement"] = conclusion
        self.sid: str = str(sid)
        self.rating: float = 0.5 if np.isnan(rating) else rating
        assert 0 <= self.rating <= 1, f"rating should be in [0,1], but was {self.rating}"
        self.weight: float = 0 if np.isnan(weight) else weight
        self.unnormalized_weight = self.weight  # this is the raw weight in [1,7] (taken from data set +1), needed for predictions
        assert 0 <= self.weight, f"weight should be positive, but was {self.weight}"
        self.username = username

    def __hash__(self):
        if self.username is not None:
            return hash(self.username)
        return id(self)

    def __eq__(self, other):
        if self.username is not None:
            return self.username == other.username
        return id(self) == id(other)

    def __repr__(self) -> str:
        return f"<{self.sid}, r={self.rating}, w={self.weight} ({self.unnormalized_weight})>"

    @cached(ttl=ttl)
    def depth(self) -> int:
        if self.conclusion is None:
            return 0
        return 1 + self.conclusion.depth()

    @cached(ttl=ttl)
    def by_name(self, name) -> Optional["Statement"]:
        """
        Find a node by statement id

        using breadth first search as we most often need nodes close to the root
        """
        if self.sid == str(name):
            return self
        queue = self.premises.copy()
        while len(queue) > 0:
            element = queue.pop()
            if element.sid == name:
                return element
            queue.extend(element.premises)
        return None

    def path_to_root(self) -> ["Statement"]:
        if self.conclusion is None:
            return [self]
        return [self] + self.conclusion.path_to_root()

    @cached(ttl=ttl)
    def product_weight_to_root(self) -> float:
        result = 1
        for statement in self.path_from_root():
            result *= statement.weight
        return result

    def path_from_root(self) -> ["Statement"]:
        return self.path_to_root()[::-1]

    def normalize_weights_to_conclusions(self) -> "Statement":
        weight_to_premises_sum = sum(p.weight for p in self.premises)
        for p in self.premises:
            if weight_to_premises_sum > 0:
                p.weight /= weight_to_premises_sum
            # else: set it to 0, but it is already 0
            p.normalize_weights_to_conclusions()
        return self


class Arguments:
    """efficient representation of premise-conclusion relations"""
    def __init__(self, arguments: pd.DataFrame):
        self.arguments = arguments.set_index("statement_id").fillna("").to_dict("index")
        for statement_id in self.arguments:
            if self.arguments[statement_id]["conclusions"] == "":
                self.arguments[statement_id]["conclusions"] = set()
            else:
                self.arguments[statement_id]["conclusions"] = {conclusion.split("(")[0] for conclusion in str(self.arguments[statement_id]["conclusions"]).split(";")}

    @cached(ttl=ttl)
    def get_positions(self) -> Set[str]:
        return {str(x) for x, v in self.arguments.items() if len(v["conclusions"]) == 0}

    @cached(ttl=ttl)
    def get_statement_ids_with_conclusion(self, conclusion_ids: Set[str]) -> Set[str]:
        """get the ids of all statements which have a conclusion which is in conclusion_ids"""
        return {str(x) for x, v in self.arguments.items() if len(v["conclusions"].intersection(conclusion_ids)) != 0}


class DeliberatePredictor(Predictor):
    """
    Predictor which uses the collaborative filtering algorithms found in deliberate using the argumentation distance
    metric by Brenneis et al.
    """

    def __init__(self, top_n: int = 100, alpha: float = 0.5, weighting_strategy: Optional[str] = "raw", max_depth: int = 2):
        """
        @param top_n: the top_n closest, relevant user profiles are considered (a profile is relevant if it has the value to be predicted)
        @param alpha: α parameter of the metric, a lower value emphasized positions
        @param weighting_strategy: raw, normalized or None: Whether the weighting uses distances normalized by the max. possible distance (raw/normalized), or do not use weighting at all (None)
        @param max_depth: maximum node depths to consider when calculating the distance between user profiles (1 only considers positions, 2 considers positions and all top-level arguments)
        """
        self.users: Dict[str, Statement] = {}
        self.arguments: Arguments = None
        self.top_n = top_n
        self.alpha = alpha
        assert weighting_strategy in {"raw", "normalized", None}
        self.weighting_strategy = weighting_strategy
        assert max_depth > 0
        self.max_depth = max_depth

    def hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        return [{"top_n": top_n, "alpha": alpha, "weighting_strategy": weighting_strategy, "max_depth": max_depth}
                for top_n in CONFIG["hyperparameter_combinations"]["deliberate"]["top_n"]
                for alpha in CONFIG["hyperparameter_combinations"]["deliberate"]["alpha"]
                for max_depth in CONFIG["hyperparameter_combinations"]["deliberate"]["max_depth"]
                for weighting_strategy in CONFIG["hyperparameter_combinations"]["deliberate"]["weighting_strategy"]]

    def predict_conviction_degree(self, statement_id: str, test_username: str) -> float:
        def distance_to_weight(distance: float) -> float:
            if self.weighting_strategy == "normalized":
                w = 1 - distance / (self.alpha * (1 - self.alpha ** self.max_depth))
                assert -0.000001 < w < 1.0000001
                return w
            elif self.weighting_strategy == "raw":
                return 1 - distance
            elif self.weighting_strategy is None:
                return 1
            raise ValueError(f"unknown weighting strategy {self.weighting_strategy}")

        this_user_graph = self.users[test_username]
        users_by_d = users_by_distance(this_user_graph, self.users, self.alpha, self.max_depth)
        relevant_users_by_d = [(graph, distance)
                               for graph, distance in users_by_d
                               if graph.by_name(statement_id) is not None and graph.by_name(statement_id).rating != 0.5][:self.top_n]
        weighted_sum = sum(distance_to_weight(distance) * graph.by_name(statement_id).rating for
                           graph, distance in relevant_users_by_d)
        normalization = sum(distance_to_weight(distance) for _, distance in relevant_users_by_d)
        if normalization == 0:
            return 0.5
        assert 0 <= weighted_sum / normalization <= 1
        return weighted_sum / normalization

    def fit(self, arguments: pd.DataFrame, training_profiles: pd.DataFrame):
        bar = ProgressBar("fitting", max=len(training_profiles))
        self.arguments = Arguments(arguments)
        self.users = ImmutableDict({profile["username"]: bar.next() or user_to_graph(self.arguments, profile) for _i, profile in training_profiles.iterrows()})
        bar.finish()

    def predict_strength(self, statement_id: str, test_username: str) -> float:
        this_user_graph = self.users[test_username]
        users_by_d = users_by_distance(this_user_graph, self.users, self.alpha, self.max_depth)[:self.top_n]
        relevant_users_by_d = [(graph, distance)
                               for graph, distance in users_by_d
                               if graph.by_name(statement_id) is not None and graph.by_name(statement_id).weight > 0]
        weighted_sum = sum((1 - distance) * (graph.by_name(statement_id).unnormalized_weight - 1)
                           for graph, distance in relevant_users_by_d)
        normalization = sum((1 - distance) for _, distance in relevant_users_by_d)
        if normalization == 0:
            return 0
        predicted_weight = weighted_sum / normalization
        assert 0 <= predicted_weight <= 6.000000001, f"predicted weight {predicted_weight} out of expected range"
        return predicted_weight


def user_to_graph(arguments: Arguments, user_data: pd.Series) -> Statement:
    """
    Transform user db entry to a weighted argumentation graph, only considering two levels of arguments (more is effectively not collected by deliberate)
    """
    def get_position_rating(pid: str):
        raw_rating = user_data.get(f"position_rating_after_{pid}", 0)
        raw_opinion = user_data.get(f"statement_attitude_{pid}", None)
        if raw_opinion is None and raw_rating != 0:
            raise ValueError(f"Got no opinion, but an rating {raw_rating} for {pid}, {user_data['username']}")
        # return np.random.randint(0, 2)
        if raw_opinion == 0:
            return 0.5 - (raw_rating+1)/7/2
        elif raw_opinion == 1:
            return (raw_rating+1)/7/2 + 0.5
        return 0.5

    I = Statement(premises=[], sid="I", weight=1, username=user_data["username"])
    position_ids = arguments.get_positions()
    assert len(position_ids) == 3, f"expected 3 positions, got {position_ids}"
    positions = [Statement(premises=[],
                           conclusion=I,
                           rating=get_position_rating(pid),
                           weight=0 if user_data.get(f"position_rating_after_{pid}") is None else 1,
                           sid=pid) for pid in position_ids]
    I.premises = positions
    for position in positions:
        premises = [Statement(premises=[],
                              conclusion=position,
                              rating=user_data.get(f"statement_attitude_{sid}", 0.5),
                              weight=user_data.get(f"argument_rating_{sid}", -1) + 1,
                              sid=sid)
                    for sid in arguments.get_statement_ids_with_conclusion({position.sid})]
        position.premises = premises
    for first_level_statement_id in arguments.get_statement_ids_with_conclusion(position_ids):
        first_level_statement = I.by_name(first_level_statement_id)
        premises = [Statement(premises=[],
                              conclusion=first_level_statement,
                              rating=user_data.get(f"statement_attitude_{sid}", 0.5),
                              weight=user_data.get(f"argument_rating_{sid}", -1) + 1,
                              sid=f"{sid}_{first_level_statement_id}")  # our weight normalization would run crazy if sids/nodes are repeated
                    for sid in arguments.get_statement_ids_with_conclusion({first_level_statement.sid})]
        first_level_statement.premises = premises
    I.normalize_weights_to_conclusions()
    return I


@cached(ttl=ttl)
def users_by_distance(this_user_graph: Statement, user_graphs: Dict[str, Statement],
                      alpha: float, max_depth: int) -> List[Tuple[Statement, float]]:
    assert this_user_graph.username != ""
    return sorted(((other_graph, d_brenneis(frozenset({this_user_graph, other_graph}), alpha=alpha, max_depth=max_depth))
                   for other_graph in user_graphs.values()
                   if other_graph.username != this_user_graph.username),
                  # for deterministic order (for reproducible results):
                  # * include nickname as tiebreaker
                  # * round float to prevent random order for "equal" values
                  key=lambda x: (round(x[1], 10), x[0].username))


@cached(ttl=ttl)
def d_brenneis(graphs: Union[Set[Statement], FrozenSet[Statement]], alpha: float, max_depth: int=2) -> float:
    """
    metric by Brenneis et al.

    @param graphs: exactly two graphs to be compared (put in set to make caching order-independent)
    @param alpha: a lower alpha emphasized opinions on arguments closer to the root (cf. PageRank)
    """
    def brenneis(g1: Statement, g2: Statement, depths_left: int) -> float:  # no need to cache recursive calls
        if (g1.weight == 0 and g2.weight == 0) or (depths_left == 0):
            return 0

        g1_weight_to_root = g1.product_weight_to_root()
        g2_weight_to_root = g2.product_weight_to_root()
        if g1_weight_to_root == 0 and g2_weight_to_root == 0:
            return 0

        first_summand = (1 - alpha) * abs((g1.rating - .5) * g1_weight_to_root - (g2.rating - .5) * g2_weight_to_root)
        second_summand = 0
        premises_ids = {p.sid for p in g1.premises + g2.premises}
        premises_1 = {p.sid: p for p in g1.premises}
        premises_2 = {p.sid: p for p in g2.premises}
        for a in premises_ids:
            a1 = premises_1.get(a)
            a2 = premises_2.get(a)
            # we used to test this, but we are quite confident that this is assured (otherwise, the construction algorithm would be broken), and it saved us 50% runtime
            # if a1 is not None and a2 is not None:
            #     assert a1.depth() == a2.depth(), "depth of nodes different, is the universe graph the same?"
            if a1 is None:  # do not use default value for get: argument construction work is bad
                a1 = Statement(sid=a)
            if a2 is None:
                a2 = Statement(sid=a)
            second_summand += brenneis(a1, a2, depths_left - 1)
        second_summand *= alpha
        return first_summand + second_summand

    assert 0 < alpha < 1
    graph1, graph2 = list(graphs)
    return brenneis(graph1, graph2, max_depth + 1)
