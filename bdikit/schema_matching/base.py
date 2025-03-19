from typing import List, NamedTuple, TypedDict, Dict
import pandas as pd


class BaseOne2oneSchemaMatcher:
    def get_one2one_match(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self, dataset: pd.DataFrame, matches: Dict[str, str]
    ) -> Dict[str, str]:
        for column in dataset.columns:
            if column not in matches:
                matches[column] = ""
        return matches


class ColumnScore(NamedTuple):
    column_name: str
    score: float


class TopkMatching(TypedDict):
    source_column: str
    top_k_columns: List[ColumnScore]


class BaseTopkSchemaMatcher(BaseOne2oneSchemaMatcher):

    def get_topk_matches(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_one2one_match(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ) -> Dict[str, str]:
        top_matches = self.get_topk_matches(source, target, 1)
        matches = {}

        for top_match in top_matches:
            source_column = top_match["source_column"]
            target_column = top_match["top_k_columns"][0].column_name
            matches[source_column] = target_column

        return matches
