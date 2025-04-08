import os
import warnings
import hashlib
import importlib
import pandas as pd
from os.path import join, dirname, isfile
from typing import Mapping, Dict, Any
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


def create_matcher(
    matcher_name: str,
    available_matchers: Dict[str, str],
    **matcher_kwargs: Mapping[str, Any],
):
    if matcher_name not in available_matchers:
        names = ", ".join(list(available_matchers.keys()))
        raise ValueError(
            f"The {matcher_name} algorithm is not supported. "
            f"Supported algorithms are: {names}"
        )

    if matcher_name == "ct_learning":
        warnings.warn(
            "ct_learning method is deprecated and will be removed in version 0.7.0 of bdi-kit. "
            "Use magneto_zs_bp, magneto_ft_bp, magneto_zs_llm or magneto_ft_llm instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
    # Load the class dynamically
    module_path, class_name = available_matchers[matcher_name].rsplit(".", 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)(**matcher_kwargs)
