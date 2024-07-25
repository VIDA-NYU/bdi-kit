import json
from os.path import join, dirname
import pandas as pd
import hashlib
import os
from bdikit.download import BDIKIT_EMBEDDINGS_CACHE_DIR

GDC_SCHEMA_PATH = join(dirname(__file__), "./resource/gdc_schema.json")
GDC_TABLE_PATH = join(dirname(__file__), "./resource/gdc_table.csv")

__gdc_df = None
__gdc_hash = None


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


def hash_dataframe(df: pd.DataFrame) -> str:

    hash_object = hashlib.sha256()

    columns_string = ",".join(df.columns) + "\n"
    hash_object.update(columns_string.encode())

    for row in df.itertuples(index=False, name=None):
        row_string = ",".join(map(str, row)) + "\n"
        hash_object.update(row_string.encode())

    return hash_object.hexdigest()


def write_embeddings_to_cache(embedding_file: str, embeddings: list):

    os.makedirs(os.path.dirname(embedding_file), exist_ok=True)

    with open(embedding_file, "w") as file:
        for vec in embeddings:
            file.write(",".join([str(val) for val in vec]) + "\n")


def load_gdc_data():
    global __gdc_df, __gdc_hash
    if __gdc_df is None or __gdc_hash is None:
        __gdc_df = pd.read_csv(GDC_TABLE_PATH)
        __gdc_hash = hash_dataframe(__gdc_df)


def get_gdc_dataframe():
    global __gdc_df
    load_gdc_data()

    return __gdc_df


def check_gdc_cache(table: pd.DataFrame, model_path: str):
    global __gdc_df, __gdc_hash
    load_gdc_data()

    table_hash = hash_dataframe(table)

    df_hash_file = None
    features = None

    # check if table for computing embedding is the same as the GDC table we have in resources
    if table_hash == __gdc_hash:
        model_name = model_path.split("/")[-1]
        cache_model_path = os.path.join(BDIKIT_EMBEDDINGS_CACHE_DIR, model_name)
        df_hash_file = os.path.join(cache_model_path, __gdc_hash)

        # Found file in cache
        if os.path.isfile(df_hash_file):
            try:
                # Load embeddings from disk
                with open(df_hash_file, "r") as file:
                    features = [
                        [float(val) for val in vec.split(",")]
                        for vec in file.read().split("\n")
                        if vec.strip()
                    ]
                    if len(features) != len(__gdc_df.columns):
                        features = None
                        raise ValueError("Mismatch in the number of features")
            except Exception as e:
                print(f"Error loading features from cache: {e}")
                features = None
    return df_hash_file, features
