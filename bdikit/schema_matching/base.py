from typing import List, NamedTuple, Any, Dict
from collections import defaultdict
import pandas as pd
import numpy as np


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
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ColumnMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self,
        source: pd.DataFrame,
        matches: List[ColumnMatch],
        default_unmatched: Any = np.nan,
    ) -> List[ColumnMatch]:
        all_source_columns = set(source.columns)

        for match in matches:
            if match.source_column in all_source_columns:
                all_source_columns.remove(match.source_column)

        # Fill missing matches with the default unmatched value
        for source_column in all_source_columns:
            matches.append(
                ColumnMatch(source_column, default_unmatched, default_unmatched)
            )

        return matches

    def _sort_matches(self, matches: List[ColumnMatch]) -> List[ColumnMatch]:
        return sorted(matches, key=lambda x: x.similarity, reverse=True)


class BaseTopkSchemaMatcher(BaseSchemaMatcher):
    def rank_schema_matches(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        top_k: int,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ColumnMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def match_schema(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ColumnMatch]:
        matches = self.rank_schema_matches(
            source, target, 1, source_context, target_context
        )

        return matches

    def _sort_ranked_matches(self, matches: List[ColumnMatch]) -> List[ColumnMatch]:
        # Group matches by source_column
        grouped_matches = defaultdict(list)
        for match in matches:
            grouped_matches[match.source_column].append(match)

        # Sort each group by similarity
        ordered_groups = [
            sorted(group, key=lambda x: x.similarity, reverse=True)
            for group in grouped_matches.values()
        ]
        # Sort the groups by maximum similarity
        ordered_groups = sorted(
            ordered_groups, key=lambda x: x[0].similarity, reverse=True
        )
        # Flatten the sorted groups into a single list
        sorted_matches = [item for group in ordered_groups for item in group]

        return sorted_matches
