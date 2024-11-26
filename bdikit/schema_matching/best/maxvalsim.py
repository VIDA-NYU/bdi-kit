import pandas as pd
from typing import Optional
from bdikit.schema_matching.best.base import BaseSchemaMatcher
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL
from bdikit.schema_matching.topk.base import BaseTopkSchemaMatcher
from bdikit.schema_matching.topk.contrastivelearning import CLTopkSchemaMatcher
from bdikit.value_matching.polyfuzz import TFIDFValueMatcher
from bdikit.value_matching.base import BaseValueMatcher


class MaxValSimSchemaMatcher(BaseSchemaMatcher):
    def __init__(
        self,
        top_k: int = 20,
        top_k_matcher: Optional[BaseTopkSchemaMatcher] = None,
        value_matcher: Optional[BaseValueMatcher] = None,
    ):
        if top_k_matcher is None:
            self.api = CLTopkSchemaMatcher(DEFAULT_CL_MODEL)
        elif isinstance(top_k_matcher, BaseTopkSchemaMatcher):
            self.api = top_k_matcher
        else:
            raise ValueError(
                f"Invalid top_k_matcher type: {type(top_k_matcher)}. "
                "Must be a subclass of {BaseTopkColumnMatcher.__name__}"
            )

        if value_matcher is None:
            self.value_matcher = TFIDFValueMatcher()
        elif isinstance(value_matcher, BaseValueMatcher):
            self.value_matcher = value_matcher
        else:
            raise ValueError(
                f"Invalid value_matcher type: {type(value_matcher)}. "
                "Must be a subclass of {BaseValueMatcher.__name__}"
            )

        self.top_k = top_k

    def unique_string_values(self, column: pd.Series) -> pd.Series:
        column = column.dropna()
        if pd.api.types.is_string_dtype(column):
            return pd.Series(column.unique(), name=column.name)
        else:
            return pd.Series(column.unique().astype(str), name=column.name)

    def map(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ):
        topk_column_matches = self.api.get_recommendations(source, target, self.top_k)

        matches = {}
        for source_column_name, scope in zip(source.columns, topk_column_matches):

            source_column_name = scope["source_column"]
            top_k_columns = scope["top_k_columns"]

            source_column = source[source_column_name]

            if not pd.api.types.is_string_dtype(source_column):
                matches[source_column_name] = top_k_columns[0].column_name
                continue

            source_values = self.unique_string_values(source_column).to_list()

            scores = []
            for top_column in top_k_columns:
                target_column_name = top_column.column_name
                target_column = target[target_column_name]
                target_values = self.unique_string_values(target_column).to_list()
                value_matches = self.value_matcher.match(source_values, target_values)
                score = sum([m.similarity for m in value_matches]) / len(target_values)
                score = (top_column.score + score) / 2.0
                scores.append((source_column_name, target_column_name, score))

            sorted_columns = sorted(scores, key=lambda it: it[2], reverse=True)

            matches[source_column_name] = sorted_columns[0][1]

        return self._fill_missing_matches(source, matches)
