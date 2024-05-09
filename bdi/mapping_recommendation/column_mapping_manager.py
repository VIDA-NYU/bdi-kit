from bdi.mapping_algorithms.column_mapping.algorithms import YurongAlgorithm


class ColumnMappingManager():

    def __init__(self, dataset, global_table):
        self.dataset = dataset
        self.global_table = global_table
        self.best_method = YurongAlgorithm()
    
    def map(self):
        mappings =  self.best_method.map()
        return mappings
