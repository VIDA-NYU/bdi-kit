import pandas as pd
from collections import defaultdict
from typing import Optional, List
from bdikit.schema_matching.base import (
    BaseOne2oneSchemaMatcher,
    BaseTopkSchemaMatcher,
    ColumnMatch,
)
from bdikit.schema_matching.valentine import SimFlood
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL
from bdikit.schema_matching.contrastivelearning import ContrastiveLearning


class TwoPhase(BaseOne2oneSchemaMatcher):
    def __init__(
        self,
        top_k: int = 20,
        top_k_matcher: Optional[BaseTopkSchemaMatcher] = None,
        schema_matcher: BaseOne2oneSchemaMatcher = SimFlood(),
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

        self.schema_matcher = schema_matcher
        self.top_k = top_k

    def get_one2one_match(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
    ) -> List[ColumnMatch]:
        topk_column_matches = self.api.get_topk_matches(source, target, self.top_k)

        grouped_matches = defaultdict(list)
        for match in topk_column_matches:
            grouped_matches[match.source_column].append(match.target_column)

        matches = []
        for source_column, candidates in grouped_matches.items():
            reduced_source = source[[source_column]]
            reduced_target = target[candidates]
            partial_matches = self.schema_matcher.get_one2one_match(
                reduced_source, reduced_target
            )
            matches += partial_matches

        matches = self._sort_matches(matches)

        return self._fill_missing_matches(source, matches)
