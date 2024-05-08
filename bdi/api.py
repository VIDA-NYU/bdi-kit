from bdi.data_ingestion.dataset_loader import load_dataframe
from bdi.mapping_recommendation.value_mapping_manager import ValueMappingManager
from bdi.mapping_recommendation.column_mapping_manager import ColumnMappingManager
from bdi.utils import get_gdc_data


class APIManager():

    def __init__(self,):
        self.dataset = None
        self.target_domain = None
        self.column_manager = None
        self.value_manager = None
        self.column_mappings = None
        self.value_mappings = None
    
    def load_dataset(self, dataset_path):
        self.dataset =  load_dataframe(dataset_path)
        self.column_manager = ColumnMappingManager(self.dataset, self.target_domain)

        return self.dataset

    def set_target_domain(self, domain='gdc'):
        pass

    def reduce_scope(self):
        pass

    def map_columns(self):
        self.column_mappings =  self.column_manager.map()

        return self.column_mappings

    def map_values(self):
        self.target_domain = get_gdc_data(self.column_mappings.values())
        self.value_manager = ValueMappingManager(self.dataset, self.column_mappings, self.target_domain )
        self.value_mappings = self.value_manager.map()

        return self.value_mappings
    

    


