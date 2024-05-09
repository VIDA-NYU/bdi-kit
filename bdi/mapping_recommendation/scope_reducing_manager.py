from bdi.mapping_algorithms.scope_reducing.algorithms import YurongReducer


class ScopeReducingManager():

    def __init__(self, dataset, target_domain):
        self.dataset = dataset
        self.target_domain = target_domain
        self.best_method = YurongReducer()
    
    def reduce(self):
        reducings = self.best_method.reduce_scope(self.dataset)
        return reducings
