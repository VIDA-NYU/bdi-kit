from bdikit import APIManager
import os


manager = APIManager()

dataset_path =  './datasets/dou.csv'
dataset = manager.load_dataset(dataset_path)
print('Dataset:')
print(dataset)
print('Reduced scope:')
reduced_scope = manager.reduce_scope()
print('Column mappings:')
column_mappings = manager.map_columns()
print('Value mappings:')
value_mappings = manager.map_values()
