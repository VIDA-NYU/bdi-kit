from enum import Enum
from os.path import join, dirname
from typing import Union, Type, List, Dict, TypedDict, Set, Optional, Tuple, Callable
import itertools
import pandas as pd
import numpy as np
from bdikit.utils import get_gdc_data
from bdikit.download import get_cached_model_or_download
from bdikit.mapping_algorithms.column_mapping.algorithms import (
    BaseColumnMappingAlgorithm,
    SimFloodAlgorithm,
    ComaAlgorithm,
    CupidAlgorithm,
    DistributionBasedAlgorithm,
    JaccardDistanceAlgorithm,
    GPTAlgorithm,
    ContrastiveLearningAlgorithm,
    TwoPhaseMatcherAlgorithm,
)
from bdikit.mapping_algorithms.value_mapping.value_mappers import ValueMapper
from bdikit.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
)
from bdikit.mapping_algorithms.value_mapping.algorithms import (
    ValueMatch,
    BaseAlgorithm,
    TFIDFAlgorithm,
    LLMAlgorithm,
    EditAlgorithm,
    EmbeddingAlgorithm,
    AutoFuzzyJoinAlgorithm,
    FastTextAlgorithm,
)
from bdikit.mapping_algorithms.value_mapping.value_mappers import (
    ValueMapper,
    FunctionValueMapper,
    DictionaryMapper,
    IdentityValueMapper,
)


GDC_DATA_PATH = join(dirname(__file__), "./resource/gdc_table.csv")


class ColumnMappingMethod(Enum):
    SIMFLOOD = ("similarity_flooding", SimFloodAlgorithm)
    COMA = ("coma", ComaAlgorithm)
    CUPID = ("cupid", CupidAlgorithm)
    DISTRIBUTION_BASED = ("distribution_based", DistributionBasedAlgorithm)
    JACCARD_DISTANCE = ("jaccard_distance", JaccardDistanceAlgorithm)
    GPT = ("gpt", GPTAlgorithm)
    CT_LEARGNING = ("ct_learning", ContrastiveLearningAlgorithm)
    TWO_PHASE = ("two_phase", TwoPhaseMatcherAlgorithm)

    def __init__(
        self, method_name: str, method_class: Type[BaseColumnMappingAlgorithm]
    ):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(method_name: str) -> BaseColumnMappingAlgorithm:
        methods = {
            method.method_name: method.method_class for method in ColumnMappingMethod
        }
        try:
            return methods[method_name]()
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )


def match_columns(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame] = "gdc",
    method: str = ColumnMappingMethod.SIMFLOOD.name,
) -> pd.DataFrame:
    """
    Performs schema mapping between the source table and the given target. The target
    either is a DataFrame or a string representing a standard data vocabulary.
    """
    if isinstance(target, str):
        target_table = _load_table_for_standard(target)
    else:
        target_table = target

    matcher_instance = ColumnMappingMethod.get_instance(method)
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
    """

    if isinstance(target, str):
        target_table = _load_table_for_standard(target)
    else:
        target_table = target

    if columns is not None and len(columns) > 0:
        selected_columns = source[columns]
    else:
        selected_columns = source

    model_path = get_cached_model_or_download("cl-reducer-v0.1")
    api = ContrastiveLearningAPI(model_path=model_path, top_k=top_k)
    _, scopes_json = api.get_recommendations(selected_columns, target=target_table)

    dfs = []
    for scope in scopes_json:
        matches = pd.DataFrame(
            scope["Top k columns"], columns=["matches", "similarity"]
        )
        matches["source"] = scope["Candidate column"]
        matches = matches[["source", "matches", "similarity"]]
        dfs.append(matches.sort_values(by="similarity", ascending=False))

    return pd.concat(dfs, ignore_index=True)


class ValueMatchingMethod(Enum):
    TFIDF = ("tfidf", TFIDFAlgorithm)
    EDIT = ("edit_distance", EditAlgorithm)
    EMBEDDINGS = ("embedding", EmbeddingAlgorithm)
    AUTOFJ = ("auto_fuzzy_join", AutoFuzzyJoinAlgorithm)
    FASTTEXT = ("fasttext", FastTextAlgorithm)
    GPT = ("gpt", LLMAlgorithm)

    def __init__(self, method_name: str, method_class: Type[BaseAlgorithm]):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(method_name: str) -> BaseAlgorithm:
        methods = {
            method.method_name: method.method_class for method in ValueMatchingMethod
        }
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
    mapper that is used to transform the values of the input column to the
    output column.
    """
    if isinstance(mapping_spec, pd.DataFrame):
        mapping_spec = mapping_spec.to_dict(orient="records")

    for mapping in mapping_spec:
        if "source" not in mapping or "target" not in mapping:
            raise ValueError(
                "Each mapping specification should contain 'source', 'target' and 'mapper' (optional) keys."
            )
        if "mapper" not in mapping:
            mapping["mapper"] = create_mapper(mapping)

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
    column_mapping: pd.DataFrame,
    method: str = ValueMatchingMethod.EDIT.name,
) -> List[ValueMatchingResult]:
    """
    Maps the values of the dataset columns to the target domain using the given method name.
    """
    if isinstance(target, str) and target == "gdc":
        column_names = column_mapping["target"].unique().tolist()
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

    column_mapping_dict = column_mapping.set_index("source")["target"].to_dict()
    value_matcher = ValueMatchingMethod.get_instance(method)
    matches = _match_values(source, target_domain, column_mapping_dict, value_matcher)
    return matches


def _match_values(
    dataset: pd.DataFrame,
    target_domain: Dict[str, Optional[List[str]]],
    column_mapping: Dict[str, str],
    value_matcher: BaseAlgorithm,
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
        target_values_dict: Dict[str, str] = {x.lower(): x for x in target_domain_list}

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


def preview_value_mappings(
    dataset: pd.DataFrame,
    column_mapping: Union[Tuple[str, str], pd.DataFrame],
    target: Union[str, pd.DataFrame] = "gdc",
    method: str = "edit_distance",
) -> List[Dict]:
    """
    Print the value mappings in a human-readable format.
    """
    if isinstance(column_mapping, pd.DataFrame):
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
            "The column_mapping must be a DataFrame or a tuple of two strings."
        )

    value_mappings = match_values(
        dataset, target=target, column_mapping=mapping_df, method=method
    )

    result = []
    for matching_result in value_mappings:

        # transform matches and unmatched values into DataFrames
        matches_df = pd.DataFrame(
            data=matching_result["matches"],
            columns=["source", "target", "similarity"],
        )

        unmatched_values = matching_result["unmatch_values"]
        unmatched_df = pd.DataFrame(
            data=list(
                zip(
                    unmatched_values,
                    [""] * len(unmatched_values),
                    [""] * len(unmatched_values),
                )
            ),
            columns=["source", "target", "similarity"],
        )

        result.append(
            {
                "source": matching_result["source"],
                "target": matching_result["target"],
                "mapping": pd.concat([matches_df, unmatched_df], ignore_index=True),
            }
        )

    if isinstance(column_mapping, tuple):
        # If only a single mapping is provided (as a tuple), we return the result
        # directly as a DataFrame to make it easier to display it in notebooks.
        assert len(result) == 1
        return result[0]["mapping"]
    else:
        return result


def preview_domains(
    dataset: pd.DataFrame,
    column_mapping: Tuple[str, str],
    target: Union[str, pd.DataFrame] = "gdc",
    limit: int = 10,
) -> pd.DataFrame:
    """
    Preview the domain of the given columns in the source and target datasets.

    Args:
        dataset (pd.DataFrame): The source dataset containing the columns to preview.
        column_mapping (Tuple[str, str]): The mapping between the source and target columns.
            The first and second positions should contain the names of the
            source and target columns respectively.
        target (Union[str, pd.DataFrame], optional): The target dataset or standard vocabulary name.
            If a string is provided and it is equal to "gdc", the target domain will be retrieved from the GDC data.
            If a DataFrame is provided, the target domain will be retrieved from the specified DataFrame.
            Defaults to "gdc".
        limit (int, optional): The maximum number of unique values to include in the preview.
            Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing a sample of the source and target domain values.
            The DataFrame will have two columns: "source_domain_sample" and "target_domain_sample".
    """
    source_column, target_column = column_mapping

    source_domain = dataset[source_column].unique()

    if isinstance(target, str) and target == "gdc":
        target_domain = np.array(get_gdc_data([target_column])[target_column])
    elif isinstance(target, pd.DataFrame):
        target_domain = target[target_column].unique()
    else:
        raise ValueError(
            "The target must be a DataFrame or a standard vocabulary name."
        )

    output_size = min(max(len(source_domain), len(target_domain)), limit)

    if len(source_domain) > output_size:
        source_domain = source_domain[:output_size]
    if len(target_domain) > output_size:
        target_domain = target_domain[:output_size]

    if len(source_domain) < output_size:
        source_domain = np.append(
            source_domain, np.full(output_size - len(source_domain), "")
        )
    if len(target_domain) < output_size:
        target_domain = np.append(
            target_domain, np.full(output_size - len(target_domain), "")
        )

    return pd.DataFrame(
        {"source_domain_sample": source_domain, "target_domain_sample": target_domain}
    )


ValueMatchingLike = Union[List[ValueMatchingResult], List[Dict], pd.DataFrame]


def update_mappings(
    value_mappings: ValueMatchingLike, user_mappings: Optional[ValueMatchingLike] = None
) -> List:

    if user_mappings is None:
        user_mappings = []

    if isinstance(value_mappings, pd.DataFrame):
        value_mappings = value_mappings.to_dict(orient="records")

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
    check_duplicates(value_mappings)

    mapping_keys = set()
    final_mappings = []

    # include all unique user mappings first, as they take precedence
    for mapping in itertools.chain(user_mappings, value_mappings):

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
        input:
            The input argument to create a ValueMapper object from.

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

            # This could be the ouput of match_values(), so can create a
            # DictionaryMapper based on the value matches
            if "matches" in input and isinstance(input["matches"], List):
                return _create_mapper_from_value_matches(input["matches"])

            # This could be the output of match_columns(), but the user did not
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
