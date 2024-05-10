from bdi.mapping_algorithms.column_mapping.algorithms import SimFlood
from enum import Enum

class MappingAlgorithm(Enum):
    YURONG = "YurongAlgorithm"
    SIMFLOOD = "SimFlood"
    
class ColumnMappingManager():

    def __init__(self, dataset, global_table, algorithm=MappingAlgorithm.SIMFLOOD):
        self._dataset = dataset #TODO: move into database object (in data_ingestion folder)
        self._global_table = global_table #TODO: move into database object (in data_ingestion folder)
        self._reduced_scope = None #TODO: move into database object (in data_ingestion folder)
        self.mapping_algorithm = algorithm 

    @property
    def reduced_scope(self):
        return self._reduced_scope

    @reduced_scope.setter
    def reduced_scope(self, value):
        self._reduced_scope = value

    @property
    def dataset(self):
        return self._dataset    
    
    @property
    def global_table(self):
        return self._global_table
    
    def map(self):
        if self.reduced_scope is None:
            mapping_algorithm_instance = eval(self.mapping_algorithm.value)(self.dataset, self.global_table)
            mappings =  mapping_algorithm_instance.map()
            return mappings
        else:
            # For each reduction suggestion, we build a new dataset and global table and run the mapping algorithm
            mappings = {}
            for reduction in self.reduced_scope:
                
                dataset_column = reduction["Candidate column"]
                global_table_columns = reduction["Top k columns"]

                if dataset_column in self.dataset.columns:
                    reduced_dataset = self.dataset[[dataset_column]]
                else:
                    continue

                common_cols = set(global_table_columns).intersection(self.global_table.columns)        
                reduced_global_table = self.global_table[list(common_cols)]
            
                mapping_algorithm_instance = eval(self.mapping_algorithm.value)(reduced_dataset, reduced_global_table)
                partial_mappings =  mapping_algorithm_instance.map()  
                
                if len(partial_mappings.keys())>0:
                    candidate_col = next(iter(partial_mappings))
                    target_col = partial_mappings[candidate_col]
                    mappings[candidate_col] = target_col
                
            return mappings
            
                
