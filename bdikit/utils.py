import os
import hashlib
import pandas as pd
from os.path import join, dirname
from bdikit.download import BDIKIT_EMBEDDINGS_CACHE_DIR
from bdikit.standards.standard_factory import Standards

GDC_TABLE_PATH = join(dirname(__file__), "./resource/gdc_table.csv")

__gdc_df = None
__gdc_hash = None


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
        standard = Standards.get_standard("gdc")
        __gdc_df = standard.get_dataframe_rep()
        __gdc_hash = hash_dataframe(__gdc_df)


def check_gdc_cache(table: pd.DataFrame, model_path: str):
    global __gdc_df, __gdc_hash
    load_gdc_data()

    table_hash = hash_dataframe(table)

    df_hash_file = None
    features = None

    # check if table for computing embedding is the same as the GDC table we have in resources
    if table_hash == __gdc_hash:
        model_name = model_path.split("/")[-1]
        cache_model_path = join(BDIKIT_EMBEDDINGS_CACHE_DIR, model_name)
        df_hash_file = join(cache_model_path, __gdc_hash)

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
