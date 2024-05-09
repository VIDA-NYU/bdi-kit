from bdi.data_ingestion.dataset_loader import load_dataframe
from bdi.mapping_recommendation.scope_reducing_manager import ScopeReducingManager
from bdi.mapping_recommendation.value_mapping_manager import ValueMappingManager
from bdi.mapping_recommendation.column_mapping_manager import ColumnMappingManager
from bdi.utils import get_gdc_data
from os.path import join, dirname

GDC_DATA_PATH = join(dirname(__file__), './resource/gdc_table.csv')
TARGET_DATA_PATH = join(dirname(__file__), './resource/target.csv')
EXAMPLE_DATA_PATH = join(dirname(__file__), './resource/cao.csv')


class APIManager():

    def __init__(self,):
        self.dataset = None
        self.global_table = None
        self.column_manager = None
        self.value_manager = None
        self.column_mappings = None
        self.value_mappings = None

    def load_global_table(self, global_table_path=None):
        if global_table_path is None:
            self.global_table = load_dataframe(GDC_DATA_PATH)
        else:
            self.global_table = load_dataframe(global_table_path)
        return self.global_table
    
    def load_dataset(self, dataset_path):
        if self.global_table is None:
            self.load_global_table()
        self.dataset =  load_dataframe(dataset_path)
        self.column_manager = ColumnMappingManager(self.dataset, self.global_table)
        return self.dataset

    def reduce_scope(self):
        self.scope_manager = ScopeReducingManager(self.dataset, self.global_table)
        return self.scope_manager.reduce()

    def map_columns(self):
        self.column_mappings =  self.column_manager.map()

        return self.column_mappings

    def map_values(self):
        self.global_table = get_gdc_data(self.column_mappings.values())
        self.value_manager = ValueMappingManager(self.dataset, self.column_mappings, self.global_table )
        self.value_mappings = self.value_manager.map()

        return self.value_mappings
    
# api = APIManager()
# dataset_path = join(dirname(__file__), './resource/dou.csv')
# dataset = api.load_dataset(dataset_path)

# print(api.dataset.head())

# print(api.global_table.head())
    

    

    

    


