from bdi import APIManager

manager = APIManager()
dataset_path =  './datasets/dou.csv'
dataset = manager.load_dataset(dataset_path)
print(dataset)
manager.set_target_domain()
manager.reduce_scope()
column_mappings = manager.map_columns()
print(column_mappings)
manager.map_values()

