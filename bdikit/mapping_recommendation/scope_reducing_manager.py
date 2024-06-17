import hashlib
import json
import os

import pandas as pd
from bdikit.download import BDIKIT_CACHE_DIR
from bdikit.mapping_algorithms.scope_reducing.algorithms import YurongReducer
from bdikit.visualization.scope_reducing import SRHeatMapManager


class ScopeReducingManager:
    def __init__(self, dataset, target_domain):
        self.dataset = dataset
        self.target_domain = target_domain
        self.best_method = YurongReducer()
        self.visualization_manager = None
        self.df_checksum = self._get_data_checksum()

    def reduce(self):
        if self._load_cached_results() is not None:
            reducings = self._load_cached_results()
        else:
            reducings = self.best_method.reduce_scope(self.dataset)
            self._cache_results(reducings)
        self.visualization_manager = SRHeatMapManager(self.dataset, reducings)
        return reducings

    def get_heatmap(self):
        self.visualization_manager.get_heatmap()
        return self.visualization_manager.plot_heatmap()

    def _get_data_checksum(self):
        return hashlib.sha1(pd.util.hash_pandas_object(self.dataset).values).hexdigest()

    def _cache_results(self, reducings):
        cache_path = os.path.join(
            BDIKIT_CACHE_DIR,
            f"reducings_{self.best_method.__class__.__name__}_{self.df_checksum}.json",
        )
        if not os.path.exists(cache_path):
            with open(cache_path, "w") as f:
                json.dump(reducings, f)

    def _load_cached_results(self):
        cache_path = os.path.join(
            BDIKIT_CACHE_DIR,
            f"reducings_{self.best_method.__class__.__name__}_{self.df_checksum}.json",
        )
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)
        return None
