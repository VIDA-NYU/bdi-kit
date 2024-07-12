from enum import Enum
from os.path import join, dirname
from typing import Union, Type, List, Dict, TypedDict, Set, Optional, Tuple, Callable
import itertools
import copy
import pandas as pd
import numpy as np
from bdikit.utils import get_gdc_data, get_gdc_metadata
from bdikit.mapping_algorithms.column_mapping.algorithms import (
    BaseSchemaMatcher,
    SimFloodSchemaMatcher,
    ComaSchemaMatcher,
    CupidSchemaMatcher,
    DistributionBasedSchemaMatcher,
    JaccardSchemaMatcher,
    GPTSchemaMatcher,
    ContrastiveLearningSchemaMatcher,
    TwoPhaseSchemaMatcher,
)
from bdikit.mapping_algorithms.value_mapping.value_mappers import ValueMapper
from bdikit.models.contrastive_learning.cl_api import (
    DEFAULT_CL_MODEL,
)
from bdikit.mapping_algorithms.column_mapping.topk_matchers import (
    CLTopkColumnMatcher,
)
from bdikit.mapping_algorithms.value_mapping.algorithms import (
    ValueMatch,
    BaseValueMatcher,
    TFIDFValueMatcher,
    GPTValueMatcher,
    EditDistanceValueMatcher,
    EmbeddingValueMatcher,
    AutoFuzzyJoinValueMatcher,
    FastTextValueMatcher,
)
from bdikit.mapping_algorithms.value_mapping.value_mappers import (
    ValueMapper,
    FunctionValueMapper,
    DictionaryMapper,
    IdentityValueMapper,
)


GDC_DATA_PATH = join(dirname(__file__), "./resource/gdc_table.csv")
DEFAULT_VALUE_MATCHING_METHOD = "tfidf"
DEFAULT_SCHEMA_MATCHING_METHOD = "coma"


class SchemaMatchers(Enum):
    SIMFLOOD = ("similarity_flooding", SimFloodSchemaMatcher)
    COMA = ("coma", ComaSchemaMatcher)
    CUPID = ("cupid", CupidSchemaMatcher)
    DISTRIBUTION_BASED = ("distribution_based", DistributionBasedSchemaMatcher)
    JACCARD_DISTANCE = ("jaccard_distance", JaccardSchemaMatcher)
    GPT = ("gpt", GPTSchemaMatcher)
    CT_LEARGNING = ("ct_learning", ContrastiveLearningSchemaMatcher)
    TWO_PHASE = ("two_phase", TwoPhaseSchemaMatcher)

    def __init__(self, method_name: str, method_class: Type[BaseSchemaMatcher]):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(method_name: str) -> BaseSchemaMatcher:
        methods = {method.method_name: method.method_class for method in SchemaMatchers}
        try:
            return methods[method_name]()
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )


def match_schema(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame] = "gdc",
    method: Union[str, BaseSchemaMatcher] = DEFAULT_SCHEMA_MATCHING_METHOD,
) -> pd.DataFrame:
    """
    Performs schema mapping between the source table and the given target schema. The
    target is either a DataFrame or a string representing a standard data vocabulary
    supported by the library. Currently, only the GDC (Genomic Data Commons) standard
    vocabulary is supported.

    Parameters:
        source (pd.DataFrame): The source table to be mapped.
        target (Union[str, pd.DataFrame], optional): The target table or standard data vocabulary. Defaults to "gdc".
        method (str, optional): The method used for mapping. Defaults to "coma".

    Returns:
        pd.DataFrame: A DataFrame containing the mapping results with columns "source" and "target".

    Raises:
        ValueError: If the method is neither a string nor an instance of BaseColumnMappingAlgorithm.
    """
    if isinstance(target, str):
        target_table = _load_table_for_standard(target)
    else:
        target_table = target

    if isinstance(method, str):
        matcher_instance = SchemaMatchers.get_instance(method)
    elif isinstance(method, BaseSchemaMatcher):
        matcher_instance = method
    else:
        raise ValueError(
            "The method must be a string or an instance of BaseColumnMappingAlgorithm"
        )

    matches = matcher_instance.map(source, target_table)

    return pd.DataFrame(matches.items(), columns=["source", "target"])


def _load_table_for_standard(name: str) -> pd.DataFrame:
    """
    Load the table for the given standard data vocabulary. Currently, only the
    GDC standard is supported.
    """
    if name == "gdc":
        return pd.read_csv(GDC_DATA_PATH)
    else:
        raise ValueError(f"The {name} standard is not supported")


def top_matches(
    source: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target: Union[str, pd.DataFrame] = "gdc",
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Returns the top-k matches between the source and target tables.

    Args:
        source (pd.DataFrame): The source table.
        columns (Optional[List[str]], optional): The list of columns to consider for matching. Defaults to None.
        target (Union[str, pd.DataFrame], optional): The target table or the name of the standard target table. Defaults to "gdc".
        top_k (int, optional): The number of top matches to return. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k matches between the source and target tables.
    """

    if isinstance(target, str):
        target_table = _load_table_for_standard(target)
    else:
        target_table = target

    if columns is not None and len(columns) > 0:
        selected_columns = source[columns]
    else:
        selected_columns = source

    topk_matcher = CLTopkColumnMatcher(model_name=DEFAULT_CL_MODEL)
    top_k_matches = topk_matcher.get_recommendations(
        selected_columns, target=target_table, top_k=top_k
    )

    dfs = []
    for match in top_k_matches:
        matches = pd.DataFrame(match["top_k_columns"], columns=["target", "similarity"])
        matches["source"] = match["source_column"]
        matches = matches[["source", "target", "similarity"]]  # reorder columns
        dfs.append(matches.sort_values(by="similarity", ascending=False))

    return pd.concat(dfs, ignore_index=True)


class ValueMatchers(Enum):
    TFIDF = ("tfidf", TFIDFValueMatcher)
    EDIT = ("edit_distance", EditDistanceValueMatcher)
    EMBEDDINGS = ("embedding", EmbeddingValueMatcher)
    AUTOFJ = ("auto_fuzzy_join", AutoFuzzyJoinValueMatcher)
    FASTTEXT = ("fasttext", FastTextValueMatcher)
    GPT = ("gpt", GPTValueMatcher)

    def __init__(self, method_name: str, method_class: Type[BaseValueMatcher]):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(method_name: str) -> BaseValueMatcher:
        methods = {method.method_name: method.method_class for method in ValueMatchers}
        try:
            return methods[method_name]()
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )


def materialize_mapping(
    input_table: pd.DataFrame, mapping_spec: Union[List[dict], pd.DataFrame]
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
        mapping_spec (Union[List[dict], pd.DataFrame]): The target mapping
          specification. It can be a list of dictionaries or a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame, which is created according to the target
        mapping specifications.
    """
    if isinstance(mapping_spec, pd.DataFrame):
        mapping_spec = mapping_spec.to_dict(orient="records")
    elif isinstance(mapping_spec, List):
        # create a shallow copy to avoid modifying the input object
        mapping_spec = [copy.copy(m) for m in mapping_spec]

    # input validation
    for mapping in mapping_spec:
        if "source" not in mapping or "target" not in mapping:
            raise ValueError(
                "Each mapping specification should contain 'source', 'target' "
                f"and 'mapper' (optional) keys but found only {mapping.keys()}."
            )

        if mapping["source"] not in input_table.columns:
            raise ValueError(
                f"The source column '{mapping['source']}' is not present in "
                " the input table."
            )

        if "mapper" not in mapping:
            mapping["mapper"] = create_mapper(mapping)

    # exectute the actual mapping plan
    output_dataframe = pd.DataFrame()
    for column_spec in mapping_spec:
        from_column_name = column_spec["source"]
        to_column_name = column_spec["target"]
        value_mapper = column_spec["mapper"]
        output_dataframe[to_column_name] = map_column_values(
            input_table[from_column_name], to_column_name, value_mapper
        )
    return output_dataframe


def map_column_values(
    input_column: pd.Series, target: str, value_mapper: ValueMapper
) -> pd.Series:
    new_column = value_mapper.map(input_column)
    new_column.name = target
    return new_column


class ValueMatchingResult(TypedDict):
    source: str
    target: str
    matches: List[ValueMatch]
    coverage: float
    unique_values: Set[str]
    unmatch_values: Set[str]


def match_values(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame],
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    method: str = DEFAULT_VALUE_MATCHING_METHOD,
) -> Union[pd.DataFrame, List[Dict]]:
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

    Returns:
        List[ValueMatchingResult]: A list of ValueMatchingResult objects
        representing the matches between the source and target values.

    Raises:
        ValueError: If the column_mapping DataFrame does not contain 'source' and
          'target' columns.
        ValueError: If the target is neither a DataFrame nor a standard vocabulary name.
        ValueError: If the source column is not present in the source dataset.
    """
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

    column_mapping_dict = mapping_df.set_index("source")["target"].to_dict()
    for source_column in column_mapping_dict.keys():
        if source_column not in source.columns:
            raise ValueError(
                f"The source column '{source_column}' is not present in the source dataset."
            )

    if isinstance(target, str) and target == "gdc":
        column_names = mapping_df["target"].unique().tolist()
        target_domain = get_gdc_data(column_names)
    elif isinstance(target, pd.DataFrame):
        target_domain = {
            column_name: target[column_name].unique().tolist()
            for column_name in target.columns
        }
    else:
        raise ValueError(
            "The target must be a DataFrame or a standard vocabulary name."
        )

    value_matcher = ValueMatchers.get_instance(method)
    matches = _match_values(source, target_domain, column_mapping_dict, value_matcher)

    result = [
        {
            "source": matching_result["source"],
            "target": matching_result["target"],
            "coverage": matching_result["coverage"],
            "matches": _value_matching_result_to_df(matching_result),
        }
        for matching_result in matches
    ]

    if isinstance(column_mapping, tuple):
        # If only a single mapping is provided (as a tuple), we return the result
        # directly as a DataFrame to make it easier to display it in notebooks.
        assert len(result) == 1
        assert isinstance(result[0]["matches"], pd.DataFrame)
        return result[0]["matches"]
    else:
        return result


def _value_matching_result_to_df(matching_result: ValueMatchingResult) -> pd.DataFrame:
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
                [None] * len(unmatched_values),
                [None] * len(unmatched_values),
            )
        ),
        columns=["source", "target", "similarity"],
    )

    return pd.concat([matches_df, unmatched_df], ignore_index=True)


def _match_values(
    dataset: pd.DataFrame,
    target_domain: Dict[str, Optional[List[str]]],
    column_mapping: Dict[str, str],
    value_matcher: BaseValueMatcher,
) -> List[ValueMatchingResult]:

    mapping_results: List[ValueMatchingResult] = []

    for source_column, target_column in column_mapping.items():

        # 1. Select candidate columns for value mapping
        target_domain_list = target_domain[target_column]
        if target_domain_list is None or len(target_domain_list) == 0:
            continue

        unique_values = dataset[source_column].unique()
        if _skip_values(unique_values):
            continue

        # 2. Transform the unique values to lowercase
        source_values_dict: Dict[str, str] = {
            str(x).strip().lower(): str(x).strip() for x in unique_values
        }
        target_values_dict: Dict[str, str] = {
            str(x).lower(): x for x in target_domain_list
        }

        # 3. Apply the value matcher to create value mapping dictionaries
        matches_lowercase = value_matcher.match(
            list(source_values_dict.keys()), list(target_values_dict.keys())
        )

        # 4. Transform the matches to the original case
        matches: List[ValueMatch] = []
        for source_value, target_value, similarity in matches_lowercase:
            matches.append(
                ValueMatch(
                    current_value=source_values_dict[source_value],
                    target_value=target_values_dict[target_value],
                    similarity=similarity,
                )
            )

        # 5. Calculate the coverage and unmatched values
        coverage = len(matches) / len(source_values_dict)
        source_values = set(source_values_dict.values())
        match_values = set([x[0] for x in matches])

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

    return mapping_results


def _skip_values(unique_values: np.ndarray, max_length: int = 50):
    if isinstance(unique_values[0], float):
        return True
    elif len(unique_values) > max_length:
        return True
    else:
        return False


def preview_domain(
    dataset: Union[str, pd.DataFrame],
    column: str,
    limit: Optional[int] = None,
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

    Returns:
        pd.DataFrame: A DataFrame containing the unique domain values (or a sample of
        them if the parameter `limit` was specified), column description and value description
        (if applicable).
    """

    if isinstance(dataset, str) and dataset == "gdc":
        gdc_metadata = get_gdc_metadata()
        value_names = gdc_metadata[column]["value_names"]
        value_descriptions = gdc_metadata[column]["value_descriptions"]
        column_description = gdc_metadata[column]["description"]
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

    domain = {"value_name": value_names}

    if len(value_descriptions) > 0:
        domain["value_description"] = value_descriptions

    if len(column_description) > 0:
        empty_rows_size = len(value_names) - 1
        domain["column_description"] = [column_description] + [""] * empty_rows_size

    return pd.DataFrame(domain)


ValueMatchingLike = Union[List[ValueMatchingResult], List[Dict], pd.DataFrame]


def update_mappings(
    mappings: ValueMatchingLike, user_mappings: Optional[ValueMatchingLike] = None
) -> List:
    """
    Creates a "data harmonization" plan based on the provided schema and/or value mappings.
    These mappings can either be computed by the library's functions or provided by the user.
    If the user mappings are provided (using the user_mappings parameter), they will take
    precedence over the mappings provided in the first parameter.

    Args:
        mappings (ValueMatchingLike): The value mappings used to create the data
            harmonization plan. It can be a pandas DataFrame or a list of dictionaries
            (ValueMatchingResult).
        user_mappings (Optional[ValueMatchingLike]): The user mappings to be included in
            the update. It can be a pandas DataFrame or a list of dictionaries (ValueMatchingResult).
            Defaults to None.

    Returns:
        List: The data harmonization plan that can be used as input to the :py:func:`~bdikit.materialize_mapping()`
        function. Concretely, the harmonization plan is a list of dictionaries, where each
        dictionary contains the source column, target column, and mapper object that will be used
        to transform the input to the output data.

    Raises:
        ValueError: If there are duplicate mappings for the same source and target columns.

    """

    if user_mappings is None:
        user_mappings = []

    if isinstance(mappings, pd.DataFrame):
        mappings = mappings.to_dict(orient="records")

    if isinstance(user_mappings, pd.DataFrame):
        user_mappings = user_mappings.to_dict(orient="records")

    def create_key(source: str, target: str) -> str:
        return source + "__" + target

    def check_duplicates(mappings: List):
        keys = set()
        for mapping in mappings:
            key = create_key(mapping["source"], mapping["target"])
            if key in keys:
                raise ValueError(
                    f"Duplicate mapping for source: {mapping['source']}, target: {mapping['target']}"
                )
            keys.add(key)

    # first check duplicates in each individual list
    check_duplicates(user_mappings)
    check_duplicates(mappings)

    mapping_keys = set()
    final_mappings = []

    # include all unique user mappings first, as they take precedence
    for mapping in itertools.chain(user_mappings, mappings):

        source_column = mapping["source"]
        target_column = mapping["target"]

        # ignore duplicate mappings accross user and value mappings
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


def create_mapper(
    input: Union[
        None,
        ValueMapper,
        pd.DataFrame,
        ValueMatchingResult,
        List[ValueMatch],
        Dict,
        Callable[[pd.Series], pd.Series],
    ]
):
    """
    Tries to instantiate an appropriate ValueMapper object for the given input argument.
    Depending on the input type, it may create one of the following objects:

    - If input is None, it creates an IdentityValueMapper object.
    - If input is a ValueMapper, it returns the input object.
    - If input is a function (or lambda function), it creates a FunctionValueMapper object.
    - If input is a list of ValueMatch objects or tuples (<source_value>, <target_value>),
      it creates a DictionaryMapper object.
    - If input is a DataFrame with two columns ("current_value", "target_value"),
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
        k in input.columns for k in ["current_value", "target_value"]
    ):
        return DictionaryMapper(
            input.set_index("current_value")["target_value"].to_dict()
        )

    if isinstance(input, Dict):
        if all(k in input for k in ["source", "target"]):
            # This could be the mapper created by update_mappings() or a
            # specification defined by the user
            if "mapper" in input:
                if isinstance(input["mapper"], ValueMapper):
                    # If it contains a ValueMapper object, just return it
                    return input["mapper"]
                else:
                    # Else, 'mapper' may contain one of the basic values that
                    # can be used to create a ValueMapper object defined above,
                    # so call this funtion recursively create it
                    return create_mapper(input["mapper"])

            # This could be the a list of value matches (i.e., ValueMatch
            # or tuple(source, target)) provided by the user
            if "matches" in input and isinstance(input["matches"], List):
                return _create_mapper_from_value_matches(input["matches"])

            if "matches" in input and isinstance(input["matches"], pd.DataFrame):
                # This could be the ouput of match_values(), so we can
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
            mapping_dict[match.current_value] = match.target_value
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
