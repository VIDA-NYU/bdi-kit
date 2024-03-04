import os
import json
import jellyfish

PATH_TO_GDC_SCHEMA = "gdc_schema.json"
def load_gdc_schema():
    if not os.path.exists(PATH_TO_GDC_SCHEMA):
        return {}
    with open(PATH_TO_GDC_SCHEMA, 'r') as f:
        data = json.load(f)
    return data

def fetch_properties(column_name):
    schema = load_gdc_schema()
    for key, values in schema.items():
        for key in values["properties"].keys():
            if column_name == key:
                return values["properties"][column_name]
            elif jellyfish.jaro_similarity(column_name, key) > 0.8:
                return values["properties"][key]
    return