import pandas as pd
from bdikit.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
)
from bdikit.download import get_cached_model_or_download


class BaseReducer:
    def __init__(self):
        pass

    def reduce_scope(self, dataset: pd.DataFrame):
        pass


class YurongReducer(BaseReducer):
    def __init__(self):
        super().__init__()
        model_path = get_cached_model_or_download("cl-reducer-v0.1")
        self.api = ContrastiveLearningAPI(model_path=model_path, top_k=20)

    def reduce_scope(self, dataset: pd.DataFrame):
        union_scopes, scopes_json = self.api.get_recommendations(dataset)
        return scopes_json
