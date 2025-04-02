import pandas as pd
from collections import defaultdict
from typing import Optional, List
from bdikit.schema_matching.base import BaseTopkSchemaMatcher, ColumnMatch
from bdikit.schema_matching.magneto import MagnetoFTBP
from bdikit.value_matching.polyfuzz import TFIDF
from bdikit.value_matching.base import BaseValueMatcher


class MaxValSim(BaseTopkSchemaMatcher):
    """
    This schema matching method first uses a a top-k column matcher (e.g., ct_learning) to prune the search space (keeping only the top-k most likely matches), and then uses a value matcher method to choose the best match from the pruned search space.
    """

    def __init__(
        self,
        top_k: int = 20,
        contribution_factor: float = 0.5,
        top_k_matcher: Optional[BaseTopkSchemaMatcher] = None,
        value_matcher: Optional[BaseValueMatcher] = None,
    ):
        if top_k_matcher is None:
            self.api = MagnetoFTBP()
        elif isinstance(top_k_matcher, BaseTopkSchemaMatcher):
            self.api = top_k_matcher
        else:
            raise ValueError(
                f"Invalid top_k_matcher type: {type(top_k_matcher)}. "
                "Must be a subclass of {BaseTopkSchemaMatcher.__name__}"
            )

        if value_matcher is None:
            self.value_matcher = TFIDF()
        elif isinstance(value_matcher, BaseValueMatcher):
            self.value_matcher = value_matcher
        else:
            raise ValueError(
                f"Invalid value_matcher type: {type(value_matcher)}. "
                "Must be a subclass of {BaseValueMatcher.__name__}"
            )

        self.top_k = top_k
        self.contribution_factor = contribution_factor

    def unique_string_values(self, column: pd.Series) -> pd.Series:
        column = column.dropna()
        if pd.api.types.is_string_dtype(column):
            return pd.Series(column.unique(), name=column.name)
        else:
            return pd.Series(column.unique().astype(str), name=column.name)

    def rank_schema_matches(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[ColumnMatch]:
        max_topk = max(
            top_k, self.top_k
        )  # If self.top_k (method param) is smaller than the requested top_k, use top_k
        topk_column_matches = self.api.rank_schema_matches(source, target, max_topk)
        top_k_results = []

        grouped_matches = defaultdict(list)
        for match in topk_column_matches:
            grouped_matches[match.source_column].append(match)

        for source_column, candidates in grouped_matches.items():
            source_values = self.unique_string_values(source[source_column]).to_list()

            scores = []
            for top_column in candidates:
                target_column_name = top_column.target_column
                if pd.isna(target_column_name):
                    continue
                target_column = target[target_column_name]
                target_values = self.unique_string_values(target_column).to_list()
                source_context = {"attribute_name": source_column}
                target_context = {"attribute_name": target_column_name}
                value_matches = self.value_matcher.match_values(
                    source_values, target_values, source_context, target_context
                )
                if len(target_values) == 0:
                    value_score = 0.0
                else:
                    value_score = sum(
                        [
                            m.similarity
                            for m in value_matches
                            if not pd.isna(m.similarity)
                        ]
                    ) / len(value_matches)

                score = (self.contribution_factor * value_score) + (
                    (1 - self.contribution_factor) * top_column.similarity
                )
                scores.append((source_column, target_column_name, score))

            sorted_columns = sorted(scores, key=lambda it: it[2], reverse=True)[:top_k]
            sorted_columns = [
                ColumnMatch(source_column, target_column, score)
                for source_column, target_column, score in sorted_columns
            ]

            top_k_results += sorted_columns

        top_k_results = self._sort_ranked_matches(top_k_results)

        return self._fill_missing_matches(source, top_k_results)
