from bdi import APIManager

manager = APIManager()

dataset_path =  './datasets/cao.csv'
dataset = manager.load_dataset(dataset_path)
print('Dataset:')
print(dataset)
manager.reduce_scope()
column_mappings = manager.map_columns()
print('Column mappings')
print(column_mappings)
value_mappings = manager.map_values()
