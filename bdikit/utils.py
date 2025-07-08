import os
import json
import pickle
import warnings
import hashlib
import inspect
import importlib
import pandas as pd
from os.path import join, dirname, isfile
from typing import Mapping, Dict, Any, List
from bdikit.download import BDIKIT_EMBEDDINGS_CACHE_DIR, BDIKIT_CACHE_DIR


def hash_dataframe(df: pd.DataFrame) -> str:
    hash_object = hashlib.sha256()

    columns_string = ",".join(df.columns) + "\n"
    hash_object.update(columns_string.encode())

    for row in df.itertuples(index=False, name=None):
        row_string = ",".join(map(str, row)) + "\n"
        hash_object.update(row_string.encode())

    return hash_object.hexdigest()


def hash_iterable(iterable):
    try:
        s = json.dumps(iterable, sort_keys=True, default=str)
    except TypeError as e:
        raise ValueError(f"Cannot serialize for hashing: {e}")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_object(obj):
    cls = obj.__class__
    init_args = {}

    try:
        sig = inspect.signature(cls.__init__)
        for name, _ in sig.parameters.items():
            if name == "self":
                continue
            # Only include args that are actually set as attributes
            if hasattr(obj, name):
                value = getattr(obj, name)
                try:
                    json.dumps(value, default=str)  # Ensure serializable
                    init_args[name] = value
                except TypeError:
                    init_args[name] = str(value)  # Fallback: stringify
    except (ValueError, TypeError):
        # Fallback: if inspection fails, fall back to empty init_args
        pass

    obj_repr = {
        "class": cls.__name__,
        "init_args": init_args,
    }

    obj_string = json.dumps(obj_repr, sort_keys=True, default=str)

    return hashlib.sha256(obj_string.encode("utf-8")).hexdigest()


def create_schema_hash(
    source_table: pd.DataFrame, target_table: pd.DataFrame, matcher: Any, **kwargs: Any
):
    source_hash = hash_dataframe(source_table)
    target_hash = hash_dataframe(target_table)
    matcher_hash = hash_object(matcher)
    topk_hash = str(kwargs.get("top_k", 1))

    final_input = json.dumps(
        {
            "source": source_hash,
            "target": target_hash,
            "matcher": matcher_hash,
            "top_k": topk_hash,
        },
        sort_keys=True,
    )
    final_hash = hashlib.sha256(final_input.encode("utf-8")).hexdigest()

    return final_hash


def create_value_hash(
    source_values: List[Any],
    target_values: List[Any],
    source_ctx: Dict[str, Any],
    target_ctx: Dict[str, Any],
    matcher: Any,
    **kwargs: Any,
):
    source_hash = hash_iterable(source_values)
    target_hash = hash_iterable(target_values)
    source_ctx_hash = hash_iterable(source_ctx)
    target_ctx_hash = hash_iterable(target_ctx)
    matcher_hash = hash_object(matcher)
    topk_hash = str(kwargs.get("top_k", 1))

    final_input = json.dumps(
        {
            "source": source_hash,
            "target": target_hash,
            "source_ctx": source_ctx_hash,
            "target_ctx": target_ctx_hash,
            "matcher": matcher_hash,
            "top_k": topk_hash,
        },
        sort_keys=True,
    )
    final_hash = hashlib.sha256(final_input.encode("utf-8")).hexdigest()

    return final_hash


def load_from_cache(hash_id: str):
    cache_path = join(BDIKIT_CACHE_DIR, "runs")
    os.makedirs(cache_path, exist_ok=True)

    hash_list = {f for f in os.listdir(cache_path) if isfile(join(cache_path, f))}

    file_path = join(cache_path, hash_id)
    file_object = None

    if hash_id in hash_list:
        if isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    file_object = pickle.load(f)

            except Exception as e:
                print(f"Error loading object from cache: {e}")

    return file_object


def save_in_cache(obj: Any, hash_id: str):
    cache_path = join(BDIKIT_CACHE_DIR, "runs")
    os.makedirs(cache_path, exist_ok=True)
    file_path = join(cache_path, hash_id)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


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
    if matcher_name == "fasttext":
        warnings.warn(
            "fasttext method is deprecated and will be removed in version 0.7.0 of bdi-kit. "
            "Use embedding instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
    # Load the class dynamically
    module_path, class_name = available_matchers[matcher_name].rsplit(".", 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)(**matcher_kwargs)


def get_class_doc(import_path: str):
    module_path, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)

    # Get the class from the module
    cls = getattr(module, class_name)

    # Get the class docstring

    doc = cls.__doc__

    if doc is None:
        return f"The class {class_name} does not have description."

    return cls.__doc__.strip()


def get_additional_context(context: Dict[str, str], dataset_label: str) -> str:
    """
    Generate a string with additional context information from the provided context dictionary.
    This function excludes the keys 'attribute_name' and 'attribute_description' from the context.
    Args:
        context (Dict[str, str]): A dictionary containing context information.
    Returns:
        str: A formatted string containing the additional context information.
    """
    str_context = ""

    for context_id in context:
        if context_id != "attribute_name" and context_id != "attribute_description":
            str_context += f"Its {context_id}: {context[context_id]}. "

    if len(str_context) > 0:
        str_context = f"The {dataset_label} dataset has the following additional context. {str_context}"

    return str_context
