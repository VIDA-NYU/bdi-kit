
import pandas as pd

from bdi.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import ContrastiveLearningAPI


class BaseReducer:

    def __init__(self):
        pass

    def reduce_scope(self, dataset: pd.DataFrame):
        pass


class YurongReducer(BaseReducer):

    def __init__(self, model_path="models/model_20_1.pt"):
        super().__init__()
        self.api = ContrastiveLearningAPI()
        
    def reduce_scope(self, dataset: pd.DataFrame):
        union_scopes, scopes_json = self.api.get_recommendations(dataset)
        return scopes_json

