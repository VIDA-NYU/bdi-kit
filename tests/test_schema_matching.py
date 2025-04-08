import pandas as pd
from bdikit.schema_matching.valentine import (
    SimFlood,
    Jaccard,
    DistributionBased,
    Coma,
    Cupid,
)

from bdikit.schema_matching.twophase import TwoPhase
from bdikit.schema_matching.contrastivelearning import ContrastiveLearning


def test_basic_schema_matching_algorithms():
    for column_matcher in [
        SimFlood(),
        Jaccard(),
        DistributionBased(),
        Coma(),
        Cupid(),
        #
        # Uncomment the following lines to test matchers that require
        # downloading large models
        #
        TwoPhase(schema_matcher=Coma()),
        ContrastiveLearning(),
    ]:
        # given
        table1 = pd.DataFrame(
            {"column_1": ["a1", "b1", "c1"], "column_2": ["a2", "b2", "c2"]}
        )
        table2 = pd.DataFrame(
            {"column_1": ["a1", "b1", "c1"], "column_2a": ["a2", "b2", "c2"]}
        )

        # when
        mapping = column_matcher.match_schema(source=table1, target=table2)

        # then
        assert ("column_1", "column_1") == (
            mapping[0].source_column,
            mapping[0].target_column,
        ), f"{type(column_matcher).__name__} failed to map columns"

        assert ("column_2", "column_2a") == (
            mapping[1].source_column,
            mapping[1].target_column,
        ), f"{type(column_matcher).__name__} failed to map columns"
