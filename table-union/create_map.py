import os
import json
import pandas as pd

input_dir = './data/'
output_dir = './table-union/'

table_map = {}

def create_table_map(file_path):
    df = pd.read_csv(file_path)
    return {i: col for i, col in enumerate(df.columns, 1)}

csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

for file_name in csv_files:
    file_path = os.path.join(input_dir, file_name)
    table_name = os.path.splitext(file_name)[0]
    table_map[table_name] = create_table_map(file_path)

json_file_path = os.path.join(output_dir, 'target_table_map.json')
with open(json_file_path, 'w') as json_file:
    json.dump(table_map, json_file, indent=4)