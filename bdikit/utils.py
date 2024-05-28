import json
from os.path import join, dirname

GDC_SCHEMA_PATH = join(dirname(__file__), "./resource/gdc_schema.json")


def read_gdc_schema():
    with open(GDC_SCHEMA_PATH) as json_file:
        gdc_schema = json.load(json_file)

    return gdc_schema


def get_gdc_data(column_names):
    gdc_schema = read_gdc_schema()
    gdc_data = {}

    for column_name in column_names:
        gdc_values = get_gdc_values(column_name, gdc_schema)
        gdc_data[column_name] = gdc_values

    return gdc_data


def get_gdc_values(column_name, gdc_schema):
    for key, values in gdc_schema.items():
        for key in values["properties"].keys():
            if column_name == key:
                value_metadata = values["properties"][column_name]
                if "enum" in value_metadata:
                    return value_metadata["enum"]
                elif "type" in value_metadata and value_metadata["type"] == "number":
                    return None

    return None


def get_all_gdc_columns():
    all_columns = []
    gdc_schema = read_gdc_schema()
    for key, values in gdc_schema.items():
        for key in values["properties"].keys():
            all_columns.append(key)
    return all_columns


def get_gdc_metadata():
    metadata = {}
    gdc_schema = read_gdc_schema()

    for key, values in gdc_schema.items():
        for key, data in values["properties"].items():
            metadata[key] = data

    return metadata
