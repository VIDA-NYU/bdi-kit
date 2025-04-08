from __future__ import annotations
import logging
import warnings
from collections import defaultdict
import itertools
import pandas as pd
import numpy as np
import panel as pn
from IPython.display import display, Markdown

from bdikit.schema_matching.base import BaseSchemaMatcher, BaseTopkSchemaMatcher
from bdikit.schema_matching.matcher_factory import (
    get_schema_matcher,
    get_topk_schema_matcher,
)
from bdikit.value_matching.base import (
    BaseValueMatcher,
    BaseTopkValueMatcher,
    ValueMatch,
    ValueMatchingResult,
)
from bdikit.value_matching.matcher_factory import (
    get_value_matcher,
    get_topk_value_matcher,
)
from bdikit.standards.standard_factory import Standards

from bdikit.mapping_functions import (
    ValueMapper,
    FunctionValueMapper,
    DictionaryMapper,
    IdentityValueMapper,
)

from typing import (
    Union,
    List,
    Dict,
    TypedDict,
    Optional,
    Tuple,
    Callable,
    Any,
)

from bdikit.config import DEFAULT_SCHEMA_MATCHING_METHOD, DEFAULT_VALUE_MATCHING_METHOD

pn.extension("tabulator")

logger = logging.getLogger(__name__)


def match_schema(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame] = "gdc",
    method: Union[str, BaseSchemaMatcher] = DEFAULT_SCHEMA_MATCHING_METHOD,
    method_args: Optional[Dict[str, Any]] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Performs schema mapping between the source table and the given target schema. The
    target is either a DataFrame or a string representing a standard data vocabulary
    supported by the library. Currently, only the GDC (Genomic Data Commons) standard
    vocabulary is supported.

    Parameters:
        source (pd.DataFrame): The source table to be mapped.
        target (Union[str, pd.DataFrame], optional): The target table or standard data vocabulary. Defaults to "gdc".
        method (str, optional): The method used for mapping. Defaults to "ct_learning".
        method_args (Dict[str, Any], optional): The additional arguments of the method for schema matching.
        standard_args (Dict[str, Any], optional): The additional arguments of the standard vocabulary.

    Returns:
        pd.DataFrame: A DataFrame containing the mapping results with columns "source" and "target".

    Raises:
        ValueError: If the method is neither a string nor an instance of BaseColumnMappingAlgorithm.
    """
    if isinstance(target, str):
        target_table = _load_table_for_standard(target, standard_args)
    else:
        target_table = target

    if isinstance(method, str):
        if method_args is None:
            method_args = {}
        matcher_instance = get_schema_matcher(method, **method_args)
    elif isinstance(method, BaseSchemaMatcher):
        matcher_instance = method
    else:
        raise ValueError(
            "The method must be a string or an instance of BaseColumnMappingAlgorithm"
        )

    matches = matcher_instance.match_schema(source, target_table)

    return pd.DataFrame(matches, columns=["source", "target", "similarity"])


def _load_table_for_standard(name: str, standard_args: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the table for the given standard data vocabulary.
    """
    if standard_args is None:
        standard_args = {}
    standard = Standards.get_standard(name, **standard_args)
    df = standard.get_dataframe_rep()

    return df


def top_matches(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame] = "gdc",
    columns: Optional[List[str]] = None,
    top_k: int = 10,
    method: Union[str, BaseTopkSchemaMatcher] = DEFAULT_SCHEMA_MATCHING_METHOD,
    method_args: Optional[Dict[str, Any]] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    .. deprecated:: 0.6.0
        **This function is deprecated, use** `rank_schema_matches` **instead**.

    Returns the top-k matches between the source and target tables.

    Args:
        source (pd.DataFrame): The source table.
        target (Union[str, pd.DataFrame], optional): The target table or the name of the standard target table. Defaults to "gdc".
        columns (Optional[List[str]], optional): The list of columns to consider for matching. Defaults to None.
        top_k (int, optional): The number of top matches to return. Defaults to 10.
        method (Union[str, BaseTopkSchemaMatcher], optional): The method used for matching. Defaults to DEFAULT_SCHEMA_MATCHING_METHOD.
        method_args (Optional[Dict[str, Any]], optional): The additional arguments of the method for schema matching.
        standard_args (Optional[Dict[str, Any]], optional): The additional arguments of the standard vocabulary.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k matches between the source and target tables.
    """
    warnings.warn(
        "`top_matches` is deprecated and will be removed in version 0.7.0 of bdi-kit. "
        "Please use `rank_schema_matches` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return rank_schema_matches(
        source=source,
        target=target,
        columns=columns,
        top_k=top_k,
        method=method,
        method_args=method_args,
        standard_args=standard_args,
    )


def rank_schema_matches(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame] = "gdc",
    columns: Optional[List[str]] = None,
    top_k: int = 10,
    method: Union[str, BaseTopkSchemaMatcher] = DEFAULT_SCHEMA_MATCHING_METHOD,
    method_args: Optional[Dict[str, Any]] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Returns the top-k matches between the source and target tables.

    Args:
        source (pd.DataFrame): The source table.
        target (Union[str, pd.DataFrame], optional): The target table or the name of the standard target table. Defaults to "gdc".
        columns (Optional[List[str]], optional): The list of columns to consider for matching. Defaults to None.
        top_k (int, optional): The number of top matches to return. Defaults to 10.
        method (Union[str, BaseTopkSchemaMatcher], optional): The method used for matching. Defaults to DEFAULT_SCHEMA_MATCHING_METHOD.
        method_args (Optional[Dict[str, Any]], optional): The additional arguments of the method for schema matching.
        standard_args (Optional[Dict[str, Any]], optional): The additional arguments of the standard vocabulary.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k matches between the source and target tables.
    """

    if isinstance(target, str):
        target_table = _load_table_for_standard(target, standard_args)
    else:
        target_table = target

    if columns is not None and len(columns) > 0:
        selected_columns = source[columns]
    else:
        selected_columns = source

    if isinstance(method, str):
        if method_args is None:
            method_args = {}
        topk_matcher = get_topk_schema_matcher(method, **method_args)
    elif isinstance(method, BaseTopkSchemaMatcher):
        topk_matcher = method
    else:
        raise ValueError(
            "The method must be a string or an instance of BaseTopkColumnMatcher"
        )

    matches = topk_matcher.rank_schema_matches(
        selected_columns, target=target_table, top_k=top_k
    )
    return pd.DataFrame(matches, columns=["source", "target", "similarity"])


def match_values(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame],
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    method: Union[str, BaseValueMatcher] = DEFAULT_VALUE_MATCHING_METHOD,
    method_args: Optional[Dict[str, Any]] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Finds matches between column values from the source dataset and column
    values of the target domain (a pd.DataFrame or a standard dictionary such
    as 'gdc') using the method provided in `method`.

    Args:
        source (pd.DataFrame): The source dataset containing the columns to be
          matched.

        target (Union[str, pd.DataFrame]): The target domain to match the
          values to. It can be either a DataFrame or a standard vocabulary name.

        column_mapping (Union[Tuple[str, str], pd.DataFrame]): A tuple or a
          DataFrame containing the mappings between source and target columns.

          - If a tuple is provided, it should contain two strings where the first
            is the source column and the second is the target column.
          - If a DataFrame is provided, it should contain 'source' and 'target'
            column names where each row specifies a column mapping.

        method (str, optional): The name of the method to use for value
          matching.
        method_args (Dict[str, Any], optional): The additional arguments of the
            method for value matching.
        standard_args (Dict[str, Any], optional): The additional arguments of the
            standard vocabulary.

    Returns:
        Union[pd.DataFrame, List[pd.DataFrame]]: A list of DataFrame objects containing
        the results of value matching between the source and target values. If a tuple
        is provided as the `column_mapping`, only a DataFrame instance is returned.

    Raises:
        ValueError: If the column_mapping DataFrame does not contain 'source' and
          'target' columns.
        ValueError: If the target is neither a DataFrame nor a standard vocabulary name.
        ValueError: If the source column is not present in the source dataset.
    """

    if standard_args is None:
        standard_args = {}

    if isinstance(method, str):
        if method_args is None:
            method_args = {}
        matcher_instance = get_value_matcher(method, **method_args)
    elif isinstance(method, BaseValueMatcher):
        matcher_instance = method

    matches = _match_values(
        source, target, column_mapping, matcher_instance, standard_args
    )

    if isinstance(column_mapping, tuple):
        if len(matches) == 0:
            return pd.DataFrame(columns=["source", "target", "similarity"])
        # If only a single mapping is provided (as a tuple), we return the result
        # directly as a DataFrame to make it easier to display it in notebooks.
        assert (
            len(matches) == 1
        ), f"Expected one result for a single column mapping, but got: {len(matches)}"
        return matches[0]
    else:
        return matches


def top_value_matches(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame],
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    top_k: int = 5,
    method: Union[str, BaseTopkValueMatcher] = DEFAULT_VALUE_MATCHING_METHOD,
    method_args: Optional[Dict[str, Any]] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> List[pd.DataFrame]:
    """
    .. deprecated:: 0.6.0
        **This function is deprecated, use** `rank_value_matches` **instead**.

    Finds top value matches between column values from the source dataset and column
    values of the target domain (a pd.DataFrame or a standard dictionary such
    as 'gdc') using the method provided in `method`.

    Args:
        source (pd.DataFrame): The source dataset containing the columns to be
          matched.

        target (Union[str, pd.DataFrame]): The target domain to match the
          values to. It can be either a DataFrame or a standard vocabulary name.

        column_mapping (Union[Tuple[str, str], pd.DataFrame]): A tuple or a
          DataFrame containing the mappings between source and target columns.

          - If a tuple is provided, it should contain two strings where the first
            is the source column and the second is the target column.
          - If a DataFrame is provided, it should contain 'source' and 'target'
            column names where each row specifies a column mapping.

        top_k (int, optional): The number of top matches to return. Defaults to 5.

        method (str, optional): The name of the method to use for value
          matching.
        method_args (Dict[str, Any], optional): The additional arguments of the
            method for value matching.
        standard_args (Dict[str, Any], optional): The additional arguments of the
            standard vocabulary.

    Returns:
        List[pd.DataFrame]: A list of DataFrame objects containing
        the results of value matching between the source and target values.

    Raises:
        ValueError: If the column_mapping DataFrame does not contain 'source' and
          'target' columns.
        ValueError: If the target is neither a DataFrame nor a standard vocabulary name.
        ValueError: If the source column is not present in the source dataset.
    """

    warnings.warn(
        "`top_value_matches` is deprecated and will be removed in version 0.7.0 of bdi-kit. "
        "Please use `rank_value_matches` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return rank_value_matches(
        source=source,
        target=target,
        column_mapping=column_mapping,
        top_k=top_k,
        method=method,
        method_args=method_args,
        standard_args=standard_args,
    )


def rank_value_matches(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame],
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    top_k: int = 5,
    method: Union[str, BaseTopkValueMatcher] = DEFAULT_VALUE_MATCHING_METHOD,
    method_args: Optional[Dict[str, Any]] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> List[pd.DataFrame]:
    """
    Finds top value matches between column values from the source dataset and column
    values of the target domain (a pd.DataFrame or a standard dictionary such
    as 'gdc') using the method provided in `method`.

    Args:
        source (pd.DataFrame): The source dataset containing the columns to be
          matched.

        target (Union[str, pd.DataFrame]): The target domain to match the
          values to. It can be either a DataFrame or a standard vocabulary name.

        column_mapping (Union[Tuple[str, str], pd.DataFrame]): A tuple or a
          DataFrame containing the mappings between source and target columns.

          - If a tuple is provided, it should contain two strings where the first
            is the source column and the second is the target column.
          - If a DataFrame is provided, it should contain 'source' and 'target'
            column names where each row specifies a column mapping.

        top_k (int, optional): The number of top matches to return. Defaults to 5.

        method (str, optional): The name of the method to use for value
          matching.
        method_args (Dict[str, Any], optional): The additional arguments of the
            method for value matching.
        standard_args (Dict[str, Any], optional): The additional arguments of the
            standard vocabulary.

    Returns:
        List[pd.DataFrame]: A list of DataFrame objects containing
        the results of value matching between the source and target values.

    Raises:
        ValueError: If the column_mapping DataFrame does not contain 'source' and
          'target' columns.
        ValueError: If the target is neither a DataFrame nor a standard vocabulary name.
        ValueError: If the source column is not present in the source dataset.
    """

    if standard_args is None:
        standard_args = {}

    if isinstance(method, str):
        if method_args is None:
            method_args = {}
        matcher_instance = get_topk_value_matcher(method, **method_args)
    elif isinstance(method, BaseTopkValueMatcher):
        matcher_instance = method

    matches = _match_values(
        source, target, column_mapping, matcher_instance, standard_args, top_k
    )

    match_list = []
    for match in matches:
        for _, group in match.groupby("source", dropna=False):
            match_list.append(
                group.reset_index(drop=True).sort_values(
                    by=["similarity"], ascending=False
                )
            )

    return match_list


def view_value_matches(
    matches: Union[pd.DataFrame, List[pd.DataFrame]], edit: bool = False
):
    """
    Shows the value match results in a DataFrame fashion.

    Args:
        matches (Union[pd.DataFrame, List[pd.DataFrame]]): The value match results
          obtained by the method match_values().

        edit (bool): Whether or not to edit the values within the DataFrame.
    """
    if isinstance(matches, pd.DataFrame):
        match_list = [matches]
    elif isinstance(matches, list):
        match_list = matches
    else:
        raise ValueError("The matches must be a DataFrame or a list of DataFrames")

    # Grouping DataFrames by metadata (source and target columns)
    grouped_matches = defaultdict(list)
    for match_df in match_list:
        grouped_matches[match_df.attrs["source"], match_df.attrs["target"]].append(
            match_df
        )

    # Display grouped DataFrames
    for (source_col, target_col), match_dfs in grouped_matches.items():
        display(
            Markdown(
                f"<br>**Source column:** {source_col}<br>"
                f"**Target column:** {target_col}<br>"
            )
        )
        for match_df in match_dfs:
            if edit:
                match_widget = pn.widgets.Tabulator(match_df, disabled=not edit)
                display(match_widget)
            else:
                display(match_df)


def _match_values(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame],
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    value_matcher: Union[BaseValueMatcher, BaseTopkValueMatcher],
    standard_args: Dict[str, Any],
    top_k: int = 1,
) -> List[pd.DataFrame]:

    target_domain, column_mapping_list = _format_value_matching_input(
        source, target, column_mapping, standard_args
    )

    mapping_results: List[ValueMatchingResult] = []

    for mapping in column_mapping_list:
        source_column, target_column = mapping["source"], mapping["target"]

        # 1. Select candidate columns for value mapping
        target_domain_list = target_domain[target_column]
        if target_domain_list is None or len(target_domain_list) == 0:
            continue

        unique_values = source[source_column].unique()
        if _skip_values(unique_values):
            continue

        # 2. Remove blank spaces to the unique values
        source_values_dict: Dict[str, List[str]] = {}
        for value in unique_values:
            stripped_value = str(value).strip()
            if stripped_value not in source_values_dict:
                source_values_dict[stripped_value] = []
            source_values_dict[stripped_value].append(value)

        target_values_dict: Dict[str, str] = {
            str(x).strip(): x for x in target_domain_list
        }

        # 3. Apply the value matcher to create value mapping dictionaries
        if isinstance(value_matcher, BaseTopkValueMatcher):
            raw_matches = value_matcher.rank_value_matches(
                list(source_values_dict.keys()), list(target_values_dict.keys()), top_k
            )
        else:
            raw_matches = value_matcher.match_values(
                list(source_values_dict.keys()), list(target_values_dict.keys())
            )

        # 4. Transform the matches to the original
        matches: List[ValueMatch] = []
        for source_value, target_value, similarity in raw_matches:
            original_source_values = source_values_dict[source_value]  # All originals
            for original_source_value in original_source_values:
                matches.append(
                    ValueMatch(
                        source_value=original_source_value,
                        target_value=target_values_dict[target_value],
                        similarity=similarity,
                    )
                )

        # 5. Calculate the coverage and unmatched values
        source_values = set(
            value for values in source_values_dict.values() for value in values
        )
        match_values = set([x[0] for x in matches])
        coverage = len(match_values) / len(source_values_dict)

        mapping_results.append(
            ValueMatchingResult(
                source=source_column,
                target=target_column,
                matches=matches,
                coverage=coverage,
                unique_values=source_values,
                unmatch_values=source_values - match_values,
            )
        )

    mapping_df_list = [
        _value_matching_result_to_df(mapping_result)
        for mapping_result in mapping_results
    ]

    return mapping_df_list


def _format_value_matching_input(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame],
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    standard_args: Dict[str, Any],
):
    if isinstance(column_mapping, pd.DataFrame):
        if not all(k in column_mapping.columns for k in ["source", "target"]):
            raise ValueError(
                "The column_mapping DataFrame must contain 'source' and 'target' columns."
            )
        mapping_df = column_mapping
    elif isinstance(column_mapping, tuple):
        mapping_df = pd.DataFrame(
            [
                {
                    "source": column_mapping[0],
                    "target": column_mapping[1],
                }
            ]
        )
    else:
        raise ValueError(
            "The column_mapping must be a DataFrame or a tuple of two strings "
            "containing the 'source' and 'target' columns."
        )

    column_mapping_list = mapping_df.to_dict(orient="records")

    for mapping in column_mapping_list:
        source_column = mapping["source"]
        if source_column not in source.columns:
            raise ValueError(
                f"The source column '{source_column}' is not present in the source dataset."
            )

    if isinstance(target, str):
        column_names = mapping_df["target"].unique().tolist()
        standard = Standards.get_standard(target, **standard_args)
        target_domain = standard.get_column_values(column_names)
    elif isinstance(target, pd.DataFrame):
        target_domain = {
            column_name: target[column_name].unique().tolist()
            for column_name in target.columns
        }
    else:
        raise ValueError(
            "The target must be a DataFrame or a standard vocabulary name."
        )

    return target_domain, column_mapping_list


def _value_matching_result_to_df(
    matching_result: ValueMatchingResult, default_unmatched: Any = np.nan
) -> pd.DataFrame:
    """
    Transforms the list of matches and unmatched values into a DataFrame.
    """
    matches_df = pd.DataFrame(
        data=matching_result["matches"],
        columns=["source", "target", "similarity"],
    )

    unmatched_values = matching_result["unmatch_values"]

    unmatched_df = pd.DataFrame(
        data=list(
            zip(
                unmatched_values,
                [default_unmatched] * len(unmatched_values),
                [default_unmatched] * len(unmatched_values),
            )
        ),
        columns=["source", "target", "similarity"],
    )

    result = pd.concat([matches_df, unmatched_df], ignore_index=True)
    result.attrs["source"] = matching_result["source"]
    result.attrs["target"] = matching_result["target"]
    result.attrs["coverage"] = matching_result["coverage"]
    return result


def _skip_values(unique_values: np.ndarray):
    if pd.api.types.is_numeric_dtype(unique_values):
        return True
    else:
        return False


def preview_domain(
    dataset: Union[str, pd.DataFrame],
    column: str,
    limit: Optional[int] = None,
    standard_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Preview the domain, i.e. set of unique values, column description and value description
    (if applicable) of the given column of the source or target dataset.

    Args:
        dataset (Union[str, pd.DataFrame], optional): The dataset or standard vocabulary name
        containing the column to preview.
            If a string is provided and it is equal to "gdc", the domain will be retrieved
            from the GDC data.
            If a DataFrame is provided, the domain will be retrieved from the specified DataFrame.
        column(str): The column name to show the domain.
        limit (int, optional): The maximum number of unique values to include in the preview.
            Defaults to None.
        standard_args (Dict[str, Any], optional): The additional arguments of the standard vocabulary.

    Returns:
        pd.DataFrame: A DataFrame containing the unique domain values (or a sample of
        them if the parameter `limit` was specified), column description and value description
        (if applicable).
    """

    if isinstance(dataset, str):
        if standard_args is None:
            standard_args = {}
        standard = Standards.get_standard(dataset, **standard_args)
        column_metadata = standard.get_column_metadata([column])
        value_names = column_metadata[column]["value_names"]
        value_descriptions = column_metadata[column]["value_descriptions"]
        column_description = column_metadata[column]["description"]
        assert len(value_names) == len(value_descriptions)
    elif isinstance(dataset, pd.DataFrame):
        value_names = dataset[column].unique()
        value_descriptions = []
        column_description = ""
    else:
        raise ValueError(
            "The dataset must be a DataFrame or a standard vocabulary name."
        )

    if isinstance(limit, int):
        value_names = value_names[:limit]
        value_descriptions = value_descriptions[:limit]

    domain = {}

    if len(value_names) > 0:
        domain["value_name"] = value_names

    if len(value_descriptions) > 0:
        domain["value_description"] = value_descriptions

    if len(column_description) > 0:
        empty_rows_size = len(value_names) - 1
        domain["column_description"] = [column_description] + [""] * empty_rows_size

    return pd.DataFrame(domain)


class ColumnMappingSpec(TypedDict):
    source: str
    target: str
    mapper: ValueMapper


def merge_mappings(
    mappings: MappingSpecLike, user_mappings: Optional[MappingSpecLike] = None
) -> List:
    """
    Creates a "data harmonization" plan based on the provided schema and/or value mappings.
    These mappings can either be computed by the library's functions or provided by the user.
    If the user mappings are provided (using the user_mappings parameter), they will take
    precedence over the mappings provided in the first parameter.

    Args:
        mappings (MappingSpecLike): The value mappings used to create the data
            harmonization plan. It can be a DataFrame, a list of dictionaries or a list of DataFrames.
        user_mappings (Optional[MappingSpecLike]): The user mappings to be included in
            the update. It can be a DataFrame, a list of dictionaries or a list of DataFrames.
            Defaults to None.

    Returns:
        List: The data harmonization plan that can be used as input to the :py:func:`~bdikit.api.materialize_mapping()`
        function. Concretely, the harmonization plan is a list of dictionaries, where each
        dictionary contains the source column, target column, and mapper object that will be used
        to transform the input to the output data.

    Raises:
        ValueError: If there are duplicate mappings for the same source and target columns.

    """
    if user_mappings is None:
        user_mappings = []

    mapping_spec_list = _normalize_mapping_spec(mappings)
    user_mapping_spec_list = _normalize_mapping_spec(user_mappings)

    def create_key(source: str, target: str) -> str:
        return source + "__" + target

    def check_duplicates(mappings: List[ColumnMappingSpec]):
        keys = set()
        for mapping in mappings:
            key = create_key(mapping["source"], mapping["target"])
            if key in keys:
                raise ValueError(
                    f"Duplicate mapping for source: {mapping['source']}, target: {mapping['target']}"
                )
            keys.add(key)

    # first check duplicates in each individual list
    check_duplicates(user_mapping_spec_list)
    check_duplicates(mapping_spec_list)

    mapping_keys = set()
    final_mappings = []

    # include all unique user mappings first, as they take precedence
    for mapping in itertools.chain(user_mapping_spec_list, mapping_spec_list):

        source_column = mapping["source"]
        target_column = mapping["target"]

        # ignore duplicate mappings across user and value mappings
        key = create_key(source_column, target_column)
        if key in mapping_keys:
            continue
        else:
            mapping_keys.add(key)

        # try creating a mapper object from the mapping
        mapper = create_mapper(mapping)

        final_mappings.append(
            {
                "source": source_column,
                "target": target_column,
                "mapper": mapper,
            }
        )

    return final_mappings


def _normalize_mapping_spec(mapping_spec: MappingSpecLike) -> List[ColumnMappingSpec]:

    if isinstance(mapping_spec, pd.DataFrame):
        mapping_spec_list: List = mapping_spec.to_dict(orient="records")
    elif isinstance(mapping_spec, List):
        mapping_spec_list: List = mapping_spec

    normalized: List[ColumnMappingSpec] = []
    for mapping_spec in mapping_spec_list:
        if isinstance(mapping_spec, pd.DataFrame):
            mapping_dict = _df_to_mapping_spec_dict(mapping_spec)
        elif isinstance(mapping_spec, Dict):
            mapping_dict = mapping_spec
        else:
            raise ValueError(
                f"Each mapping specification must be a dictionary or a DataFrame,"
                f" but was: {str(mapping_spec)}"
            )

        if "source" not in mapping_dict or "target" not in mapping_dict:
            raise ValueError(
                "Each mapping specification should contain 'source', 'target' "
                f"and 'mapper' (optional) keys but found only {mapping_dict.keys()}."
            )

        if "mapper" in mapping_dict and isinstance(mapping_dict["mapper"], ValueMapper):
            mapper = mapping_dict["mapper"]
        else:
            mapper = create_mapper(mapping_dict)

        normalized.append(
            {
                "source": mapping_dict["source"],
                "target": mapping_dict["target"],
                "mapper": mapper,
            }
        )

    return normalized


def _df_to_mapping_spec_dict(spec: Union[Dict, pd.DataFrame]) -> Dict:
    if isinstance(spec, Dict):
        return spec
    elif isinstance(spec, pd.DataFrame):
        if "source" not in spec.attrs or "target" not in spec.attrs:
            raise ValueError(
                "The DataFrame must contain 'source' and 'target' attributes."
            )
        return {
            "source": spec.attrs["source"],
            "target": spec.attrs["target"],
            "matches": spec,
        }
    else:
        raise ValueError(f"Invalid mapping specification: {str(spec)}")


def materialize_mapping(
    input_table: pd.DataFrame, mapping_spec: MappingSpecLike
) -> pd.DataFrame:
    """
    Takes an input DataFrame and a target mapping specification and returns a
    new DataFrame created according to the given target mapping specification.
    The mapping specification is a list of dictionaries, where each dictionary
    defines one column in the output table and how it is created. It includes
    the names of the input (source) and output (target) columns and the value
    mapper used to transform the values of the input column into the
    target output column.

    Parameters:
        input_table (pd.DataFrame): The input (source) DataFrame.
        mapping_spec (MappingSpecLike): The target mapping specification. It can
            be a DataFrame, a list of dictionaries or a list of DataFrames.

    Returns:
        pd.DataFrame: A DataFrame, which is created according to the target
        mapping specifications.
    """
    mapping_spec_list = _normalize_mapping_spec(mapping_spec)

    for mapping in mapping_spec_list:
        if mapping["source"] not in input_table.columns:
            raise ValueError(
                f"The source column '{mapping['source']}' is not present in "
                " the input table."
            )

    # execute the actual mapping plan
    output_dataframe = pd.DataFrame()
    for column_spec in mapping_spec_list:
        from_column_name = column_spec["source"]
        to_column_name = column_spec["target"]
        value_mapper = column_spec["mapper"]
        output_dataframe[to_column_name] = value_mapper.map(
            input_table[from_column_name]
        )
    return output_dataframe


def create_mapper(
    input: Union[
        None,
        ValueMapper,
        pd.DataFrame,
        ValueMatchingResult,
        List[ValueMatch],
        Dict,
        ColumnMappingSpec,
        Callable[[pd.Series], pd.Series],
    ],
):
    """
    Tries to instantiate an appropriate ValueMapper object for the given input argument.
    Depending on the input type, it may create one of the following objects:

    - If input is None, it creates an IdentityValueMapper object.
    - If input is a ValueMapper, it returns the input object.
    - If input is a function (or lambda function), it creates a FunctionValueMapper object.
    - If input is a list of ValueMatch objects or tuples (<source_value>, <target_value>),
      it creates a DictionaryMapper object.
    - If input is a DataFrame with two columns ("source_value", "target_value"),
      it creates a DictionaryMapper object.
    - If input is a dictionary containing a "source" and "target" key, it tries to create
      a ValueMapper object based on the specification given in "mapper" or "matches" keys.

    Args:
        input: The input argument to create a ValueMapper object from.

    Returns:
        ValueMapper: An instance of a ValueMapper.
    """
    # If no input is provided, we create an IdentityValueMapper by default
    # to not change the values from the source column
    if input is None:
        return IdentityValueMapper()

    # If the input is already a ValueMapper, no need to create a new one
    if isinstance(input, ValueMapper):
        return input

    # If the input is a function, we can create a FunctionValueMapper
    # that applies the function to the values of the source column
    if callable(input):
        return FunctionValueMapper(input)

    # This could be a list of value matches produced by match_values(),
    # so can create a DictionaryMapper based on the value matches
    if isinstance(input, List):
        return _create_mapper_from_value_matches(input)

    # If the input is a DataFrame with two columns, we can create a
    # DictionaryMapper based on the values in the DataFrame
    if isinstance(input, pd.DataFrame) and all(
        k in input.columns for k in ["source_value", "target_value"]
    ):
        return DictionaryMapper(
            input.set_index("source_value")["target_value"].to_dict()
        )

    if isinstance(input, Dict):
        if all(k in input for k in ["source", "target"]):
            # This could be the mapper created by merge_mappings() or a
            # specification defined by the user
            if "mapper" in input:
                if isinstance(input["mapper"], ValueMapper):
                    # If it contains a ValueMapper object, just return it
                    return input["mapper"]
                else:
                    # Else, 'mapper' may contain one of the basic values that
                    # can be used to create a ValueMapper object defined above,
                    # so call this function recursively create it
                    return create_mapper(input["mapper"])

            # This could be the a list of value matches (i.e., ValueMatch
            # or tuple(source, target)) provided by the user
            if "matches" in input and isinstance(input["matches"], List):
                return _create_mapper_from_value_matches(input["matches"])

            if "matches" in input and isinstance(input["matches"], pd.DataFrame):
                # This could be the output of match_values(), so we can
                # create a DictionaryMapper based on the value matches
                return DictionaryMapper(
                    input["matches"].set_index("source")["target"].to_dict()
                )

            # This could be the output of match_schema(), but the user did not
            # define any mapper, so we create an IdentityValueMapper to map the
            # column to the target name but keeping the values as they are
            return IdentityValueMapper()

    raise ValueError(f"Failed to create a ValueMapper for given input: {input}")


def _create_mapper_from_value_matches(matches: List[ValueMatch]) -> DictionaryMapper:
    mapping_dict = {}
    for match in matches:
        if isinstance(match, ValueMatch):
            mapping_dict[match.source_value] = match.target_value
        elif isinstance(match, tuple) and len(match) == 2:
            if isinstance(match[0], str) and isinstance(match[1], str):
                mapping_dict[match[0]] = match[1]
            else:
                raise ValueError(
                    "Tuple in matches must contain two strings: (source_value, target_value)"
                )
        else:
            raise ValueError("Matches must be a list of ValueMatch objects or tuples")
    return DictionaryMapper(mapping_dict)


MappingSpecLike = Union[List[Union[Dict, pd.DataFrame]], pd.DataFrame]
"""
The `MappingSpecLike` is a type alias that specifies mappings between source
and target columns. It must include the source and target column names
and a value mapper object that transforms the values of the source column
into the target.

The mapping specification can be (1) a DataFrame or (2) a list of dictionaries or DataFrames.

If it is a list of dictionaries, they must have:

- `source`: The name of the source column.
- `target`: The name of the target column.
- `mapper` (optional): A ValueMapper instance or an object that can be used to
  create one using :py:func:`~bdikit.api.create_mapper()`. Examples of valid objects
  are Python functions or lambda functions. If empty, an IdentityValueMapper
  is used by default.
- `matches` (optional): Specifies the value mappings. It can be a DataFrame containing
  the matches (returned by :py:func:`~bdikit.api.match_values()`), a list of ValueMatch
  objects, or a list of tuples (<source_value>, <target_value>).

Alternatively, the list can contain DataFrames. In this case, the DataFrames must
contain not only the value mappings (as described in the `matches` key above) but
also the `source` and `target` columns as DataFrame attributes. The DataFrames created
by :py:func:`~bdikit.api.match_values()` include this information by default.

If the mapping specification is a DataFrame, it must be compatible with the dictionaries
above and contain `source`, `target`, and `mapper` or `matcher` columns.

Example:

.. code-block:: python

    mapping_spec = [
      {
        "source": "source_column1",
        "target": "target_column1",
      },
      {
        "source": "source_column2",
        "target": "target_column2",
        "mapper": lambda age: -age * 365.25,
      },
      {
        "source": "source_column3",
        "target": "target_column3",
        "matches": [
          ("source_value1", "target_value1"),
          ("source_value2", "target_value2"),
        ]
      },
      {
        "source": "source_column",
        "target": "target_column",
        "matches": df_value_mapping_1
      },
      df_value_mapping_2, # a DataFrame returned by match_values()
    ]
"""
