import os
import pandas as pd
from bdi.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import \
    ContrastiveLearningAPI
from os.path import join, dirname

MODEL_PATH = join(dirname(__file__), "../../../resource/model_20_1.pt")


class BaseReducer:
    def __init__(self):
        pass

    def reduce_scope(self, dataset: pd.DataFrame):
        pass


class YurongReducer(BaseReducer):

    def __init__(self):
        super().__init__()
        model_path = os.environ.get('BDI_MODEL_PATH', MODEL_PATH)
        self.api = ContrastiveLearningAPI(model_path=model_path)

    def reduce_scope(self, dataset: pd.DataFrame):
        union_scopes, scopes_json = self.api.get_recommendations(dataset)
        return scopes_json
