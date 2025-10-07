import random
import bdikit as bdi
import pandas as pd
from pydantic import BaseModel
from os.path import join, exists
from bdikit.utils import get_class_doc
from mcp.server.fastmcp import FastMCP
from bdikit.schema_matching.matcher_factory import schema_matchers, topk_schema_matchers
from bdikit.value_matching.matcher_factory import topk_value_matchers, value_matchers
from typing import Any, Optional, Dict, List, Union


class AttributeMatch(BaseModel):
    source_attribute: str
    target_attribute: str


# Initialize FastMCP server
server = FastMCP(
    name="BDI-Kit",
    instructions="This MCP server provides tools from BDI-Kit for data integration and harmonization. "
    "BDI-Kit offers a diverse suite of harmonization methods, each tailored to excel in specific scenarios. "
    "You can use the tools to perform schema matching, value matching, and data materialization tasks. "
    "Schema matching is the process of identifying corresponding attributes/columns between two datasets. "
    "Value matching is the process of identifying corresponding values between two attributes/columns. "
    "Data materialization is the process of saving the source dataset formatted according to the target format using the found matches.",
)
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
    target_dataset_path = target_dataset_path or "gdc"
    method = method or "magneto_ft_bp"

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
    target_dataset_path = target_dataset_path or "gdc"
    method = method or "magneto_ft_bp"

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
    attribute_matches: Optional[List[AttributeMatch]] = None,
    method: Optional[str] = "tfidf",
) -> List[Dict[str, Any]]:
    """Finds matches between attribute/column values from the source dataset and attribute/column
    values of the target schema.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Path to target schema or a standard vocabulary name (e.g. "gdc", which uses the GDC schema)
        attribute_matches: The attribute/column pairs from the source and target dataset for which to find value matches.
            If not provided, it will use  the attribute matches from the schema matching step.
        method: Optional method to use for value matching (default is "tfidf").

    Returns:
        Dictionary with value matching results
    """
    method = method or "tfidf"
    source_dataset = pd.read_csv(source_dataset_path)

    if target_dataset_path == "gdc":
        target_dataset = target_dataset_path
    else:
        target_dataset = pd.read_csv(target_dataset_path)

    if attribute_matches is None:
        if server.schema_matching_results is not None:
            attribute_matches = server.schema_matching_results.to_dict(orient="records")
        else:
            raise ValueError(
                "No schema match results available. Please provide attribute_matches or run match_schema first."
            )

    value_matches_list = []
    for match in attribute_matches:
        source = match["source_attribute"]
        target = match["target_attribute"]

        matches = bdi.match_values(
            source_dataset,
            target_dataset,
            (source, target),
            method=method,
        )
        value_matches_list.append(matches)

    value_matches = pd.concat(value_matches_list, ignore_index=True)

    if server.value_matching_results is None:
        server.value_matching_results = pd.DataFrame()

    server.value_matching_results = pd.concat(
        [server.value_matching_results, value_matches], ignore_index=True
    )

    response = value_matches.to_dict(orient="records")

    return response


@server.tool()
async def rank_value_matches(
    source_dataset_path: str,
    target_dataset_path: str,
    attribute_matches: Optional[List[List[str]]] = None,
    top_k: Optional[int] = 5,
    method: Optional[str] = "tfidf",
) -> List[Dict[str, Any]]:
    """Returns the top-k value matches between the source and target attributes/columns. Where k is a value specified by the user.

    Args:
        source_dataset_path: Path to the source CSV data file
        target_dataset_path: Path to target schema or a standard vocabulary name (e.g. "gdc", which uses the GDC schema)
        attribute_matches: The attribute/column of the source and target dataset for which to find value matches for.
            If not provided, it will use  the attribute matches from the schema matching step.
        top_k: Optional number of top matches to return (default is 5)
        method: Optional method to use for value matching (default is "tfidf").

    Returns:
        Dictionary with value matching results
    """
    method = method or "tfidf"
    source_dataset = pd.read_csv(source_dataset_path)

    if target_dataset_path == "gdc":
        target_dataset = target_dataset_path
    else:
        target_dataset = pd.read_csv(target_dataset_path)

    if attribute_matches is None:
        if server.schema_matching_results is not None:
            attribute_matches = server.schema_matching_results.to_dict(orient="records")
        else:
            raise ValueError(
                "No schema match results available. Please provide attribute_matches or run match_schema first."
            )

    value_matches_list = []
    for match in attribute_matches:
        source = match["source_attribute"]
        target = match["target_attribute"]

        matches = bdi.rank_value_matches(
            source_dataset,
            target_dataset,
            (source, target),
            top_k=top_k,
            method=method,
        )
        value_matches_list.append(matches)

    response = pd.concat(value_matches_list, ignore_index=True).to_dict(
        orient="records"
    )

    return response


@server.tool()
async def preview_domain(
    dataset_path: str,
    attribute: str,
) -> List[Dict[str, Any]]:
    """
    Preview the domain, attribute description and values description
    (if applicable) of the given attribute of the source or target dataset.

    Args:
        dataset_path (str): The path to the dataset file (CSV) or standard vocabulary name
        containing the attribute to preview.
            If a string is provided and it is equal to "gdc", the domain will be retrieved
            from the GDC data.
            If a path is provided, the domain will be retrieved from the specified path.
        attribute(str): The attribute name to show the domain.

    Returns:
        Dictionary with the description of the  description and value description
        (if applicable).
    """
    if exists(dataset_path):
        dataset = pd.read_csv(dataset_path)
    else:
        dataset = dataset_path
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
async def update_value_matching(
    source_attribute: str,
    target_attribute: str,
    source_value: Union[str, float],
    new_target_value: Union[str, float],
    new_similarity: float,
) -> List[Dict[str, Any]]:
    """Updates the value matching results for a specific source attribute, target attribute, and source value with a new target value and similarity score.
    Args:
        source_attribute: The source attribute to update
        target_attribute: The target attribute to update
        source_value: The source value to update
        new_target_value: The new target value to set for the source value
        new_similarity: The new similarity score to set for the new pair
    """
    if server.value_matching_results is None:
        raise ValueError(
            "No value match results available. Please run match_values first."
        )

    # Update the value match results with the new value matches
    server.value_matching_results.loc[
        (server.value_matching_results["source_attribute"] == source_attribute)
        & (server.value_matching_results["target_attribute"] == target_attribute)
        & (server.value_matching_results["source_value"] == source_value),
        "target_value",
    ] = new_target_value

    server.value_matching_results.loc[
        (server.value_matching_results["source_attribute"] == source_attribute)
        & (server.value_matching_results["target_attribute"] == target_attribute)
        & (server.value_matching_results["source_value"] == source_value),
        "similarity",
    ] = new_similarity

    response = server.value_matching_results.to_dict(orient="records")

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


def run_mcp_server():
    server.run(transport="stdio")
