import pandas as pd
from typing import List, Callable
from bdikit.schema_matching.base import BaseSchemaMatcher, ColumnMatch
from valentine import valentine_match
from valentine.algorithms.matcher_results import MatcherResults
from valentine.algorithms.jaccard_distance import StringDistanceFunction
from valentine.algorithms import (
    SimilarityFlooding as SimilarityFloodingMatcher,
    Coma as ComaMatcher,
    Cupid as CupidMatcher,
    DistributionBased as DistributionBasedMatcher,
    JaccardDistanceMatcher,
    BaseMatcher,
)


class Valentine(BaseSchemaMatcher):
    def __init__(self, matcher: BaseMatcher):
        self.matcher = matcher

    def match_schema(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> List[ColumnMatch]:
        raw_matches: MatcherResults = valentine_match(source, target, self.matcher)
        matches = []
        cache_dict = set()  # To guarantee the one-to-one matching
        for match_data, score in raw_matches.one_to_one().items():
            source_column = match_data[0][1]
            target_column = match_data[1][1]
            if source_column not in cache_dict:
                matches.append(ColumnMatch(source_column, target_column, score))
                cache_dict.add(source_column)

        matches = self._sort_matches(matches)

        return self._fill_missing_matches(source, matches)


class SimFlood(Valentine):
    """Similarity Flooding transforms schemas into directed graphs and merges them into a propagation graph. The algorithm iteratively propagates similarity scores to neighboring nodes until convergence."""

    def __init__(
        self, coeff_policy: str = "inverse_average", formula: str = "formula_c"
    ):
        super().__init__(
            SimilarityFloodingMatcher(coeff_policy=coeff_policy, formula=formula)
        )


class Coma(Valentine):
    """COMA is a matcher that combines multiple schema-based matchers, representing schemas as rooted directed acyclic graphs."""

    def __init__(
        self, max_n: int = 0, use_instances: bool = False, java_xmx: str = "1024m"
    ):
        super().__init__(
            ComaMatcher(max_n=max_n, use_instances=use_instances, java_xmx=java_xmx)
        )


class Cupid(Valentine):
    """Cupid calculates overall similarity using linguistic and structural similarities, with tree transformations helping to compute context-based similarity."""

    def __init__(
        self,
        leaf_w_struct: float = 0.2,
        w_struct: float = 0.2,
        th_accept: float = 0.7,
        th_high: float = 0.6,
        th_low: float = 0.35,
        c_inc: float = 1.2,
        c_dec: float = 0.9,
        th_ns: float = 0.7,
        parallelism: int = 1,
    ):
        super().__init__(
            CupidMatcher(
                leaf_w_struct=leaf_w_struct,
                w_struct=w_struct,
                th_accept=th_accept,
                th_high=th_high,
                th_low=th_low,
                c_inc=c_inc,
                c_dec=c_dec,
                th_ns=th_ns,
                parallelism=parallelism,
            )
        )


class DistributionBased(Valentine):
    """Distribution-based Matching compares the distribution of data values in columns using the Earth Mover’s Distance. It clusters relational attributes based on these comparisons."""

    def __init__(
        self,
        threshold1: float = 0.15,
        threshold2: float = 0.15,
        quantiles: int = 256,
        process_num: int = 1,
    ):
        super().__init__(
            DistributionBasedMatcher(
                threshold1=threshold1,
                threshold2=threshold2,
                quantiles=quantiles,
                process_num=process_num,
            )
        )


class Jaccard(Valentine):
    """
    This algorithm computes pairwise column similarities using Jaccard similarity, treating values as identical if their Levenshtein distance is below a threshold.
    """

    def __init__(
        self,
        threshold_dist: float = 0.8,
        distance_fun: Callable[[str, str], float] = StringDistanceFunction.Levenshtein,
        process_num: int = 1,
    ):
        super().__init__(
            JaccardDistanceMatcher(
                threshold_dist=threshold_dist,
                distance_fun=distance_fun,
                process_num=process_num,
            )
        )
