import numbers
import random
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
server.schema_matching_results = None
server.value_matching_results = None

random.seed(42)  # For reproducibility


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

    server.schema_matching_results = matches
    response = matches.to_dict(orient="records")

    return response


@server.tool()
async def rank_schema_matches(
    source_dataset_path: str,
    target_dataset_path: Optional[str] = "gdc",
    attributes: Optional[List[str]] = None,
    top_k: Optional[int] = 5,
    method: Optional[str] = "magneto_ft_bp",
) -> List[Dict[str, Any]]:
    """Returns the top-k matches between the source and target tables. Where k is a value specified by the user.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Optional path to target schema (default is "gdc", which uses the GDC schema)
        attributes: Optional list of attributes/columns to match
        top_k: Optional number of top matches to return (default is 5)
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
    method: Optional[str] = "auto",
) -> List[Dict[str, Any]]:
    """Finds matches between attribute/column values from the source dataset and attribute/column
    values of the target schema.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Path to target schema or a standard vocabulary name (e.g. "gdc", which uses the GDC schema)
        attribute_matches: The attribute/column pairs from the source and target dataset for which to find value matches.
            If not provided, it will be calculated using all attributes/columns.
        method: Optional method to use for value matching. Default is "auto", which selects the method based on the data type of the values.

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

    if method == "auto":
        # Automatically select the method based on the data type of the values
        source_values = source_dataset[attribute_matches[0]].unique().tolist()
        method = select_method(source_values)

    matches = bdi.match_values(
        source_dataset,
        target_dataset,
        attribute_matches,
        method=method,
    )

    if server.value_matching_results is None:
        server.value_matching_results = pd.DataFrame()

    server.value_matching_results = pd.concat(
        [server.value_matching_results, matches], ignore_index=True
    )

    response = matches.to_dict(orient="records")

    return response


@server.tool()
async def rank_value_matches(
    source_dataset_path: str,
    target_dataset_path: str,
    attribute_matches: Union[Tuple[str, str], None],
    top_k: Optional[int] = 5,
    method: Optional[str] = "auto",
) -> List[Dict[str, Any]]:
    """Returns the top-k value matches between the source and target attributes/columns. Where k is a value specified by the user.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Path to target schema or a standard vocabulary name (e.g. "gdc", which uses the GDC schema)
        attribute_matches: The attribute/column of the source and target dataset for which to find value matches for.
            If not provided, it will be calculated using all attributes/columns.
        top_k: Optional number of top matches to return (default is 5)
        method: Optional method to use for value matching. Default is "auto", which selects the method based on the data type of the values.

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
    if method == "auto":
        # Automatically select the method based on the data type of the values
        source_values = source_dataset[attribute_matches[0]].unique().tolist()
        method = select_method(source_values)

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
async def preview_domain(
    dataset: Union[str, pd.DataFrame],
    attribute: str,
) -> pd.DataFrame:
    """
    Preview the domain, attribute description and values description
    (if applicable) of the given attribute of the source or target dataset.

    Args:
        dataset (Union[str, pd.DataFrame], optional): The dataset or standard vocabulary name
        containing the attribute to preview.
            If a string is provided and it is equal to "gdc", the domain will be retrieved
            from the GDC data.
            If a DataFrame is provided, the domain will be retrieved from the specified DataFrame.
        attribute(str): The attribute name to show the domain.

    Returns:
        Dictionary with the description of the  description and value description
        (if applicable).
    """
    domain = bdi.preview_domain(dataset, attribute)
    response = domain.to_dict(orient="records")

    return response


@server.tool()
async def materialize_mapping(
    source_dataset_path: str,
    output_folder_path: str,
    file_name: Optional[str] = "materialized_data.csv",
) -> List[Dict[str, Any]]:
    """Takes the source dataset, the output folder path, and an optional file name,
        and materializes the data, saving it as a CSV file in the specified output folder.

    Args:
        source_dataset_path: Path to the source CSV data file
        output_folder_path: Path to the folder where the materialized data will be saved
        file_name: Optional name for the output CSV file (default is "materialized_data.csv")
    Returns:
        Dictionary with a message of the materialization result and the materialized data
    """
    source_dataset = pd.read_csv(source_dataset_path)

    if server.value_matching_results is not None:
        mapping_spec_df = server.value_matching_results
    elif server.schema_matching_results is not None:
        mapping_spec_df = server.schema_matching_results
    else:
        raise ValueError(
            "No schema or value matching results available. Please run match_schema or match_values first."
        )

    materialized_data = bdi.materialize_mapping(source_dataset, mapping_spec_df)
    output_file_path = join(output_folder_path, file_name)
    materialized_data.to_csv(output_file_path, index=False)

    response = {}
    response["message"] = f"Materialized data saved successfully in {output_file_path}."
    response["data"] = materialized_data.to_dict(orient="records")

    return response


@server.tool()
async def update_schema_matching(
    source_attribute: str, new_target_attribute: str, new_similarity: float
) -> List[Dict[str, Any]]:
    """Updates the schema matching results for a specific source attribute with a new target attribute and similarity score.
    Args:
        source_attribute: The source attribute to update
        new_target_attribute: The new target attribute to set for the source attribute
        new_similarity: The new similarity score to set for the new pair
    Returns:
        List of dictionaries with the updated schema matching results.
    """
    if server.schema_matching_results is None:
        raise ValueError(
            "No schema match results available. Please run match_schema first."
        )

    # Update the schema match results with the new attribute matches
    server.schema_matching_results.loc[
        server.schema_matching_results["source_attribute"] == source_attribute,
        "target_attribute",
    ] = new_target_attribute
    server.schema_matching_results.loc[
        server.schema_matching_results["source_attribute"] == source_attribute,
        "similarity",
    ] = new_similarity

    response = server.schema_matching_results.to_dict(orient="records")

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


def determine_type(values, threshold=0.7, sample_size=10):
    numeric_count = 0

    if len(values) > sample_size:
        sample_values = random.sample(values, sample_size)
    else:
        sample_values = values

    for value in sample_values:
        # Try to convert strings to numbers
        if isinstance(value, numbers.Number):
            numeric_count += 1
        else:
            try:
                float(value)  # Attempt to cast to a float
                numeric_count += 1
            except (ValueError, TypeError):
                continue

    proportion_numeric = numeric_count / len(sample_values)

    if proportion_numeric >= threshold:
        return "numeric"
    else:
        return "text"


def select_method(values):
    value_type = determine_type(values)
    if value_type == "numeric":
        return "llm_numeric"
    else:
        return "tfidf"
