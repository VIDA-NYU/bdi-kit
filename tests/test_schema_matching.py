import pandas as pd
from bdikit.schema_matching.best.valentine import (
    SimFloodSchemaMatcher,
    JaccardSchemaMatcher,
    DistributionBasedSchemaMatcher,
    ComaSchemaMatcher,
    CupidSchemaMatcher,
)
from bdikit.schema_matching.best.twophase import TwoPhaseSchemaMatcher
from bdikit.schema_matching.best.contrastivelearning import ContrastiveLearningSchemaMatcher


def test_basic_column_mapping_algorithms():
    for column_matcher in [
        SimFloodSchemaMatcher(),
        JaccardSchemaMatcher(),
        DistributionBasedSchemaMatcher(),
        ComaSchemaMatcher(),
        CupidSchemaMatcher(),
        #
        # Uncomment the following lines to test matchers that require
        # downloading large models
        #
        TwoPhaseSchemaMatcher(schema_matcher=ComaSchemaMatcher()),
        ContrastiveLearningSchemaMatcher(),
    ]:
        # given
        table1 = pd.DataFrame(
            {"column_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]}
        )
        table2 = pd.DataFrame(
            {"column_1a": ["a1", "b1", "c1"], "col2": ["a2", "b2", "c2"]}
        )

        # when
        mapping = column_matcher.map(source=table1, target=table2)

        # then
        assert {
            "column_1": "column_1a",
            "col_2": "col2",
        } == mapping, f"{type(column_matcher).__name__} failed to map columns"
