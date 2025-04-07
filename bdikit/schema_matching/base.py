from typing import List, NamedTuple
from collections import defaultdict
import pandas as pd


class ColumnMatch(NamedTuple):
    """
    Represents a match between a source column and a target column with a
    similarity score.
    """

    source_column: str
    target_column: str
    similarity: float


class BaseSchemaMatcher:
    def match_schema(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> List[ColumnMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self, dataset: pd.DataFrame, matches: List[ColumnMatch]
    ) -> List[ColumnMatch]:
        all_source_columns = set(dataset.columns)

        for match in matches:
            if match.source_column in all_source_columns:
                all_source_columns.remove(match.source_column)

        # Fill missing matches with empty strings
        for source_column in all_source_columns:
            matches.append(ColumnMatch(source_column, "", ""))

        return matches

    def _sort_matches(self, matches: List[ColumnMatch]) -> List[ColumnMatch]:
        return sorted(matches, key=lambda x: x.similarity, reverse=True)


class BaseTopkSchemaMatcher(BaseSchemaMatcher):

    def rank_schema_matches(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[ColumnMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def match_schema(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ) -> List[ColumnMatch]:
        matches = self.rank_schema_matches(source, target, 1)

        return matches

    def _sort_ranked_matches(self, matches: List[ColumnMatch]) -> List[ColumnMatch]:
        # Step 1: Group matches by source_column
        grouped_matches = defaultdict(list)
        for match in matches:
            grouped_matches[match.source_column].append(match)

        # Step 2: Sort each group internally by similarity (descending)
        for source_col in grouped_matches:
            grouped_matches[source_col].sort(key=lambda x: x.similarity, reverse=True)

        # Step 3: Sort groups by the highest similarity in each group, then flatten the result
        sorted_matches = sorted(
            (
                match for group in grouped_matches.values() for match in group
            ),  # Flatten groups
            key=lambda x: grouped_matches[x.source_column][
                0
            ].similarity,  # Sort by highest similarity per group
            reverse=True,
        )

        return sorted_matches
