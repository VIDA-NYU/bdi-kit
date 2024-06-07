import unittest
import pandas as pd
from bdikit.mapping_algorithms.column_mapping.algorithms import (
    SimFloodAlgorithm,
    JaccardDistanceAlgorithm,
    DistributionBasedAlgorithm,
    ComaAlgorithm,
    CupidAlgorithm,
)


class ColumnMappingTest(unittest.TestCase):
    def test_basic_column_mapping_algorithms(self):
        for ColumnMatcher in [
            SimFloodAlgorithm,
            JaccardDistanceAlgorithm,
            DistributionBasedAlgorithm,
            ComaAlgorithm,
            CupidAlgorithm,
        ]:
            # given
            table1 = pd.DataFrame(
                {"column_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]}
            )
            table2 = pd.DataFrame(
                {"column_1a": ["a1", "b1", "c1"], "col2": ["a2", "b2", "c2"]}
            )
            column_matcher = ColumnMatcher()

            # when
            mapping = column_matcher.map(dataset=table1, global_table=table2)

            # then
            print(mapping)
            self.assertEqual(
                {"column_1": "column_1a", "col_2": "col2"},
                mapping,
                msg=f"{ColumnMatcher.__name__} failed to map columns",
            )
