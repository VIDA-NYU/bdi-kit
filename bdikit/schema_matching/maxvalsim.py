import pandas as pd
from typing import Optional, List
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL
from bdikit.schema_matching.base import (
    BaseTopkSchemaMatcher,
    TopkMatching,
    ColumnScore,
)
from bdikit.schema_matching.contrastivelearning import ContrastiveLearning
from bdikit.value_matching.polyfuzz import TFIDF
from bdikit.value_matching.base import BaseOne2oneValueMatcher


class MaxValSim(BaseTopkSchemaMatcher):
    def __init__(
        self,
        top_k: int = 20,
        contribution_factor: float = 0.5,
        top_k_matcher: Optional[BaseTopkSchemaMatcher] = None,
        value_matcher: Optional[BaseOne2oneValueMatcher] = None,
    ):
        if top_k_matcher is None:
            self.api = ContrastiveLearning(DEFAULT_CL_MODEL)
        elif isinstance(top_k_matcher, BaseTopkSchemaMatcher):
            self.api = top_k_matcher
        else:
            raise ValueError(
                f"Invalid top_k_matcher type: {type(top_k_matcher)}. "
                "Must be a subclass of {BaseTopkColumnMatcher.__name__}"
            )

        if value_matcher is None:
            self.value_matcher = TFIDF()
        elif isinstance(value_matcher, BaseOne2oneValueMatcher):
            self.value_matcher = value_matcher
        else:
            raise ValueError(
                f"Invalid value_matcher type: {type(value_matcher)}. "
                "Must be a subclass of {BaseOne2oneValueMatcher.__name__}"
            )

        self.top_k = top_k
        self.contribution_factor = contribution_factor

    def unique_string_values(self, column: pd.Series) -> pd.Series:
        column = column.dropna()
        if pd.api.types.is_string_dtype(column):
            return pd.Series(column.unique(), name=column.name)
        else:
            return pd.Series(column.unique().astype(str), name=column.name)

    def get_topk_matches(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        max_topk = max(
            top_k, self.top_k
        )  # If self.top_k (method param) is smaller than the requested top_k, use top_k
        topk_column_matches = self.api.get_topk_matches(source, target, max_topk)
        matches = {}
        top_k_results = []

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
                value_matches = self.value_matcher.get_one2one_match(
                    source_values, target_values
                )
                if len(target_values) == 0:
                    value_score = 0.0
                else:
                    value_score = sum([m.similarity for m in value_matches]) / len(
                        target_values
                    )

                score = (self.contribution_factor * value_score) + (
                    (1 - self.contribution_factor) * top_column.score
                )
                scores.append((source_column_name, target_column_name, score))

            sorted_columns = sorted(scores, key=lambda it: it[2], reverse=True)[:top_k]
            sorted_columns = [
                ColumnScore(name, score) for _, name, score in sorted_columns
            ]

            top_k_results.append(
                {
                    "source_column": source_column_name,
                    "top_k_columns": sorted_columns,
                }
            )

        return top_k_results
