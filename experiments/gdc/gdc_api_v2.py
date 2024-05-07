import json
import logging
import os
from os.path import dirname, join
import pandas as pd

import jellyfish

logger = logging.getLogger(__name__)

PATH_TO_GDC_SCHEMA = join(dirname(__file__), "gdc_schema.json")


class GDCSchema:
    def __init__(self, subschemas=None):
        self.schema = load_gdc_schema(subschemas)
        self.subschemas = subschemas

    def parse_schema_to_df(self):
        ret = {}
        for parent, values in self.schema.items():
            data_dict = {"column_name": [], "column_type": [], "column_description": [], "column_values": []}
            for candidate in values["properties"].keys():
                if self.get_column_type(values["properties"][candidate]) is None:
                    continue
                data_dict["column_name"].append(candidate)
                data_dict["column_type"].append(self.get_column_type(values["properties"][candidate]))
                data_dict["column_description"].append(self.get_column_description(values["properties"][candidate]))
                data_dict["column_values"].append(self.get_column_values(values["properties"][candidate]))
            ret[parent] = pd.DataFrame(data_dict)
        return ret
    
    def get_properties_by_column_name(self, column_name):
        properties = []
        for parent, values in self.schema.items():
            for candidate in values["properties"].keys():
                if candidate == column_name:
                    properties.append(candidate)
                    properties.append(self.get_column_type(values["properties"][candidate]))
                    properties.append(self.get_column_description(values["properties"][candidate]))
                    properties.append(self.get_column_values(values["properties"][candidate]))
                    return properties
        return [None, None, None, []]

    def get_column_type(self, properties):
        if "enum" in properties:
            return "enum"
        elif "type" in properties:
            return properties["type"]
        else:
            return None
    
    def get_column_description(self, properties):
        if "description" in properties:
            return properties["description"]
        elif "common" in properties:
            return properties["common"]["description"]
        return ""
    
    def get_column_values(self, properties):
        col_type = self.get_column_type(properties)
        if col_type == "enum":
            return properties["enum"]
        elif col_type == "number" or col_type == "integer" or col_type == "float":
            return [
                str(properties["minimum"])
                if "minimum" in properties
                else "-inf",
                str(properties["maximum"])
                if "maximum" in properties
                else "inf",
            ]
        elif col_type == "boolean":
            return ["True", "False"]
        else:
            return None


def load_gdc_schema(subschemas=None):
    if not os.path.exists(PATH_TO_GDC_SCHEMA):
        return {}
    with open(PATH_TO_GDC_SCHEMA, "r") as f:
        data = json.load(f)

    if subschemas:
        return {subschema: data[subschema] for subschema in subschemas}
    else:
        return data