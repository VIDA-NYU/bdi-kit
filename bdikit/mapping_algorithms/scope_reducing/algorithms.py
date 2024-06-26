import pandas as pd
from bdikit.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
    DEFAULT_CL_MODEL,
    GDC_TABLE_PATH,
)


class BaseReducer:
    def __init__(self):
        pass

    def reduce_scope(self, dataset: pd.DataFrame):
        pass


class YurongReducer(BaseReducer):
    def __init__(self):
        super().__init__()
        self.api = ContrastiveLearningAPI(model_name=DEFAULT_CL_MODEL)

    def reduce_scope(self, dataset: pd.DataFrame):
        gdc_ds = pd.read_csv(GDC_TABLE_PATH)
        _, scopes_json = self.api.get_recommendations(dataset, target=gdc_ds, top_k=20)
        return scopes_json
