from bdi.mapping_algorithms.column_mapping.algorithms import YurongAlgorithm


class ColumnMappingManager():

    def __init__(self, dataset, target_domain):
        self.dataset = dataset
        self.target_domain = target_domain
        self.best_method = YurongAlgorithm()
    
    def map(self):
        mappings =  self.best_method.map()
        return mappings
