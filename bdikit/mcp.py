import bdikit as bdi
import pandas as pd
from os.path import join
from bdikit.utils import get_class_doc
from mcp.server.fastmcp import FastMCP
from bdikit.schema_matching.matcher_factory import schema_matchers, topk_schema_matchers
from bdikit.value_matching.matcher_factory import topk_value_matchers, value_matchers
from typing import Any, Optional, Dict, List, Union, Tuple


# Initialize FastMCP server
server = FastMCP("bdikit-harmonizer")


@server.tool()
async def match_schema(
    source_dataset_path: str,
    target_dataset_path: Optional[str] = "gdc",
    method: Optional[str] = "magneto_ft_bp",
) -> List[Dict[str, Any]]:
    """Performs schema matching task between the source table and the given target schema

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Optional path to target schema (default is "gdc", which uses the GDC schema)
        method: Optional method to use for schema matching (default is "magneto_ft_bp")

    Returns:
        Dictionary with schema matching results
    """
    source_dataset = pd.read_csv(source_dataset_path)

    if target_dataset_path == "gdc":
        target_dataset = target_dataset_path
    else:
        target_dataset = pd.read_csv(target_dataset_path)

    matches = bdi.match_schema(source_dataset, target=target_dataset, method=method)

    response = matches.to_dict(orient="records")

    return response


@server.tool()
async def rank_schema_matches(
    source_dataset_path: str,
    target_dataset_path: Optional[str] = "gdc",
    attributes: Optional[List[str]] = None,
    top_k: Optional[int] = 10,
    method: Optional[str] = "magneto_ft_bp",
) -> List[Dict[str, Any]]:
    """Returns the top-k matches between the source and target tables. Where k is a value specified by the user.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Optional path to target schema (default is "gdc", which uses the GDC schema)
        attributes: Optional list of attributes/columns to match
        top_k: Optional number of top matches to return (default is 10)
        method: Optional method to use for schema matching (default is "magneto_ft_bp")

    Returns:
        Dictionary with schema matching results
    """
    source_dataset = pd.read_csv(source_dataset_path)

    if target_dataset_path == "gdc":
        target_dataset = target_dataset_path
    else:
        target_dataset = pd.read_csv(target_dataset_path)

    matches = bdi.rank_schema_matches(
        source_dataset,
        target=target_dataset,
        attributes=attributes,
        top_k=top_k,
        method=method,
    )

    response = matches.to_dict(orient="records")

    return response


@server.tool()
async def match_values(
    source_dataset_path: str,
    target_dataset_path: str,
    attribute_matches: Union[Tuple[str, str], None],
    method: Optional[str] = "tfidf",
) -> List[Dict[str, Any]]:
    """Finds matches between attribute/column values from the source dataset and attribute/column
    values of the target schema.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Path to target schema or a standard vocabulary name (e.g. "gdc", which uses the GDC schema)
        attribute_matches: The attribute/column of the source and target dataset for which to find value matches for.
            If not provided, it will be calculated using all attributes/columns.
        method: Optional method to use for value matching (default is "tf-idf")

    Returns:
        Dictionary with value matching results
    """

    source_dataset = pd.read_csv(source_dataset_path)

    if target_dataset_path == "gdc":
        target_dataset = target_dataset_path
    else:
        target_dataset = pd.read_csv(target_dataset_path)

    if attribute_matches is None:
        # Match all attributes if no specific matches are provided
        attribute_matches = bdi.match_schema(
            source_dataset,
            target=target_dataset,
        )

    matches = bdi.match_values(
        source_dataset,
        target_dataset,
        attribute_matches,
        method=method,
    )

    response = matches.to_dict(orient="records")

    return response


@server.tool()
async def rank_value_matches(
    source_dataset_path: str,
    target_dataset_path: str,
    attribute_matches: Union[Tuple[str, str], None],
    top_k: Optional[int] = 5,
    method: Optional[str] = "tfidf",
) -> List[Dict[str, Any]]:
    """Returns the top-k value matches between the source and target attributes/columns. Where k is a value specified by the user.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Path to target schema or a standard vocabulary name (e.g. "gdc", which uses the GDC schema)
        attribute_matches: The attribute/column of the source and target dataset for which to find value matches for.
            If not provided, it will be calculated using all attributes/columns.
        top_k: Optional number of top matches to return (default is 5)
        method: Optional method to use for value matching (default is "tf-idf")

    Returns:
        Dictionary with value matching results
    """

    source_dataset = pd.read_csv(source_dataset_path)

    if target_dataset_path == "gdc":
        target_dataset = target_dataset_path
    else:
        target_dataset = pd.read_csv(target_dataset_path)

    if attribute_matches is None:
        # Match all attributes if no specific matches are provided
        attribute_matches = bdi.match_schema(
            source_dataset,
            target=target_dataset,
        )

    matches = bdi.rank_value_matches(
        source_dataset,
        target_dataset,
        attribute_matches,
        top_k=top_k,
        method=method,
    )

    response = matches.to_dict(orient="records")

    return response


@server.tool()
async def materialize_mapping(
    source_dataset_path: str,
    mapping_spec: List[Dict[str, Any]],
    output_folder_path: str,
    file_name: Optional[str] = "materialized_data.csv",
) -> List[Dict[str, Any]]:
    """Takes the source dataset, the mapping specification, the output folder path, and an optional file name,
        and materializes the data according to the mapping specification, saving it as a CSV file in the specified output folder.

    Args:
        source_dataset_path: Path to the source CSV data file
        mapping_spec: List of dictionaries representing the mapping specification. For instance, the output of the match_values() function:
            [{'source_attribute': '', 'target_attribute': 'source_value': '','target_value': '', 'similarity': 1}, ...]
        output_folder_path: Path to the folder where the materialized data will be saved
        file_name: Optional name for the output CSV file (default is "materialized_data.csv")
    Returns:
        Dictionary with a message of the materialization result and the materialized data
    """
    source_dataset = pd.read_csv(source_dataset_path)

    # Convert to DataFrame
    mapping_spec_df = pd.DataFrame(mapping_spec)
    materialized_data = bdi.materialize_mapping(source_dataset, mapping_spec_df)
    output_file_path = join(output_folder_path, file_name)
    materialized_data.to_csv(output_file_path, index=False)

    response = {}
    response["message"] = f"Materialized data saved successfully in {output_file_path}."
    response["data"] = materialized_data.to_dict(orient="records")

    return response


@server.tool()
async def get_available_schema_matching_algorithms(
    mode: str = "top_1",
) -> List[Dict[str, str]]:
    """Returns the available schema matching algorithms.
    Args:
        mode: The mode of schema matching algorithms to return. Can be "top_1" for single value matchers or "top_k" for top-k value matchers.
            Default is "top_1".
    Returns:
        List of dictionaries with available schema matching algorithms. Each dictionary contains the name and description of the algorithm.
    """

    if mode == "top_1":
        available_algorithms = schema_matchers
    elif mode == "top_k":
        available_algorithms = topk_schema_matchers
    else:
        raise ValueError(
            "Invalid mode. Use 'top_1' for single value matchers or 'top_k' for top-k value matchers."
        )

    response = [
        {"name": name, "description": get_class_doc(path)}
        for name, path in available_algorithms.items()
    ]
    return response


@server.tool()
async def get_available_value_matching_algorithms(
    mode: str = "top_1",
) -> List[Dict[str, str]]:
    """Returns the available value matching algorithms.
    Args:
        mode: The mode of value matching algorithms to return. Can be "top_1" for single value matchers or "top_k" for top-k value matchers.
            Default is "top_1".
    Returns:
        List of dictionaries with available value matching algorithms. Each dictionary contains the name and description of the algorithm.
    """

    if mode == "top_1":
        available_algorithms = value_matchers
    elif mode == "top_k":
        available_algorithms = topk_value_matchers
    else:
        raise ValueError(
            "Invalid mode. Use 'top_1' for single value matchers or 'top_k' for top-k value matchers."
        )

    response = [
        {"name": name, "description": get_class_doc(path)}
        for name, path in available_algorithms.items()
    ]
    return response
