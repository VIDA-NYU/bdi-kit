from bdikit.schema_matching.one2one.base import BaseSchemaMatcher
from typing import List, NamedTuple, TypedDict, Dict
import pandas as pd


class ColumnScore(NamedTuple):
    column_name: str
    score: float


class TopkMatching(TypedDict):
    source_column: str
    top_k_columns: List[ColumnScore]


class BaseTopkSchemaMatcher(BaseSchemaMatcher):

    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        raise NotImplementedError("Subclasses must implement this method")

    def map(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ) -> Dict[str, str]:
        top_matches = self.get_recommendations(source, target, 1)
        matches = {}

        for top_match in top_matches:
            source_column = top_match["source_column"]
            target_column = top_match["top_k_columns"][0].column_name
            matches[source_column] = target_column

        return matches
