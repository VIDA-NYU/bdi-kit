from bdi import APIManager
import os

os.environ['BDI_MODEL_PATH'] = 'YOUR PATH HERE'

manager = APIManager()

dataset_path =  './datasets/dou.csv'
dataset = manager.load_dataset(dataset_path)
print('Dataset:')
print(dataset)
reduced_scope = manager.reduce_scope()
print('Reduced scope:')
print(reduced_scope)
column_mappings = manager.map_columns()
print('Column mappings:')
print(column_mappings)
value_mappings = manager.map_values()
