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


def get_gdc_metadata():
    metadata = {}
    gdc_schema = read_gdc_schema()

    for attrib_data in gdc_schema.values():
        for attrib_name, attrib_properties in attrib_data["properties"].items():
            metadata[attrib_name] = {}
            attrib_description = attrib_properties.get("description", "")
            metadata[attrib_name]["description"] = attrib_description

            value_names = attrib_properties.get("enum", [])
            metadata[attrib_name]["value_names"] = value_names

            descriptions = attrib_properties.get("enumDef", {})
            value_descriptions = []
            for value_name in value_names:
                description = ""
                if value_name in descriptions:
                    description = descriptions[value_name].get("description", "")
                value_descriptions.append(description)

            metadata[attrib_name]["value_descriptions"] = value_descriptions

    return metadata


def get_gdc_layered_metadata():
    metadata = {}
    gdc_schema = read_gdc_schema()

    for subschema, values in gdc_schema.items():
        for key, data in values["properties"].items():
            metadata[key] = (subschema, data)

    return metadata
