import os
import hashlib
import pandas as pd
from os.path import join, dirname, isfile
from bdikit.download import BDIKIT_EMBEDDINGS_CACHE_DIR


def hash_dataframe(df: pd.DataFrame) -> str:
    hash_object = hashlib.sha256()

    columns_string = ",".join(df.columns) + "\n"
    hash_object.update(columns_string.encode())

    for row in df.itertuples(index=False, name=None):
        row_string = ",".join(map(str, row)) + "\n"
        hash_object.update(row_string.encode())

    return hash_object.hexdigest()


def write_embeddings_to_cache(embedding_file: str, embeddings: list):

    os.makedirs(dirname(embedding_file), exist_ok=True)

    with open(embedding_file, "w") as file:
        for vec in embeddings:
            file.write(",".join([str(val) for val in vec]) + "\n")


def check_embedding_cache(table: pd.DataFrame, model_path: str):
    embedding_file = None
    embeddings = None
    table_hash = hash_dataframe(table)
    model_name = model_path.split("/")[-1]
    cache_model_path = join(BDIKIT_EMBEDDINGS_CACHE_DIR, model_name)
    os.makedirs(cache_model_path, exist_ok=True)

    hash_list = {
        f for f in os.listdir(cache_model_path) if isfile(join(cache_model_path, f))
    }

    embedding_file = join(cache_model_path, table_hash)

    # Check if table for computing embedding is the same as the tables we have in resources
    if table_hash in hash_list:
        if isfile(embedding_file):
            try:
                # Load embeddings from disk
                with open(embedding_file, "r") as file:
                    embeddings = [
                        [float(val) for val in vec.split(",")]
                        for vec in file.read().split("\n")
                        if vec.strip()
                    ]

            except Exception as e:
                print(f"Error loading features from cache: {e}")
                embeddings = None

    return embedding_file, embeddings
