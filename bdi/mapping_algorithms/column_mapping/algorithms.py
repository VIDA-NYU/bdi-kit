from valentine import valentine_match
from valentine.algorithms import SimilarityFlooding,Coma,Cupid,DistributionBased,JaccardDistanceMatcher


class BaseColumnMappingAlgorithm():
    
    def __init__(self, dataset, global_table):
        self._dataset = dataset
        self._global_table = global_table

    def map(self):
        raise NotImplementedError("Subclasses must implement this method")


class SimFlood(BaseColumnMappingAlgorithm):

    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
    
    def map(self):
        matcher = SimilarityFlooding()
        matches = valentine_match(self._dataset, self._global_table, matcher)

        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate        
        return mappings
    
class ComaAlgorithm(BaseColumnMappingAlgorithm):

    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
    
    def map(self):
        matcher = Coma()
        matches = valentine_match(self._dataset, self._global_table, matcher)

        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate        
        return mappings
    
class CupidAlgorithm(BaseColumnMappingAlgorithm):
    
        def __init__(self, dataset, global_table):
            super().__init__(dataset, global_table)
        
        def map(self):
            matcher = Cupid()
            matches = valentine_match(self._dataset, self._global_table, matcher)
    
            mappings = {}
            for match in matches.one_to_one():
                dataset_candidate = match[0][1]
                global_table_candidate = match[1][1]
                mappings[dataset_candidate] = global_table_candidate        
            return mappings
        
class DistributionBasedAlgorithm(BaseColumnMappingAlgorithm):
        
            def __init__(self, dataset, global_table):
                super().__init__(dataset, global_table)
            
            def map(self):
                matcher = DistributionBased()
                matches = valentine_match(self._dataset, self._global_table, matcher)
        
                mappings = {}
                for match in matches.one_to_one():
                    dataset_candidate = match[0][1]
                    global_table_candidate = match[1][1]
                    mappings[dataset_candidate] = global_table_candidate        
                return mappings
            
class JaccardDistanceAlgorithm(BaseColumnMappingAlgorithm):
                
                def __init__(self, dataset, global_table):
                    super().__init__(dataset, global_table)
                
                def map(self):
                    matcher = JaccardDistanceMatcher()
                    matches = valentine_match(self._dataset, self._global_table, matcher)
            
                    mappings = {}
                    for match in matches.one_to_one():
                        dataset_candidate = match[0][1]
                        global_table_candidate = match[1][1]
                        mappings[dataset_candidate] = global_table_candidate        
                    return mappings
            

