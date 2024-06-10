from bdikit.mapping_algorithms.scope_reducing.algorithms import YurongReducer
from bdikit.visualization.scope_reducing import SRHeatMapManager


class ScopeReducingManager:
    def __init__(self, dataset, target_domain):
        self.dataset = dataset
        self.target_domain = target_domain
        self.best_method = YurongReducer()
        self.visualization_manager = None

    def reduce(self):
        reducings = self.best_method.reduce_scope(self.dataset)
        self.visualization_manager = SRHeatMapManager(self.dataset, reducings)
        return reducings

    def get_heatmap(self):
        self.visualization_manager.get_heatmap()
        return self.visualization_manager.plot_heatmap()
