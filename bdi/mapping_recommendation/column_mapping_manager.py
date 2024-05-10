from bdi.mapping_algorithms.column_mapping.algorithms import SimFlood
from enum import Enum

class MappingAlgorithm(Enum):
    YURONG = "YurongAlgorithm"
    SIMFLOOD = "SimFlood"
    
class ColumnMappingManager():

    def __init__(self, dataset, global_table, algorithm=MappingAlgorithm.SIMFLOOD):
        self.dataset = dataset
        self.global_table = global_table
        self.mapping_algorithm = eval(algorithm.value)(dataset, global_table)
    
    def map(self):
        mappings =  self.mapping_algorithm.map()
        return mappings
