import pandas as pd
from typing import Optional
from bdikit.schema_matching.best.base import BaseSchemaMatcher
from bdikit.schema_matching.best.valentine import SimFloodSchemaMatcher
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL
from bdikit.schema_matching.topk.base import BaseTopkSchemaMatcher
from bdikit.schema_matching.topk.contrastivelearning import CLTopkSchemaMatcher


class TwoPhaseSchemaMatcher(BaseSchemaMatcher):
    def __init__(
        self,
        top_k: int = 20,
        top_k_matcher: Optional[BaseTopkSchemaMatcher] = None,
        schema_matcher: BaseSchemaMatcher = SimFloodSchemaMatcher(),
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

        self.schema_matcher = schema_matcher
        self.top_k = top_k

    def map(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ):
        topk_column_matches = self.api.get_recommendations(source, target, self.top_k)

        matches = {}
        for column, scope in zip(source.columns, topk_column_matches):
            candidates = [
                cand[0] for cand in scope["top_k_columns"] if cand[0] in target.columns
            ]
            reduced_source = source[[column]]
            reduced_target = target[candidates]
            partial_matches = self.schema_matcher.map(reduced_source, reduced_target)

            if column in partial_matches:
                matches[column] = partial_matches[column]

        return self._fill_missing_matches(source, matches)
