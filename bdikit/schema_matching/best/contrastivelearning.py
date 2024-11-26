import pandas as pd
from bdikit.schema_matching.best.base import BaseSchemaMatcher
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL
from bdikit.schema_matching.topk.contrastivelearning import CLTopkSchemaMatcher


class ContrastiveLearningSchemaMatcher(BaseSchemaMatcher):
    def __init__(self, model_name: str = DEFAULT_CL_MODEL):
        self.topk_matcher = CLTopkSchemaMatcher(model_name=model_name)

    def map(self, source: pd.DataFrame, target: pd.DataFrame):
        topk_matches = self.topk_matcher.get_recommendations(source, target, top_k=1)
        matches = {}
        for column, top_k_match in zip(source.columns, topk_matches):
            candidate = top_k_match["top_k_columns"][0][0]
            if candidate in target.columns:
                matches[column] = candidate
        return self._fill_missing_matches(source, matches)
