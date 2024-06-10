from enum import Enum
from os.path import join, dirname
from typing import Union, Type, List, Optional
import pandas as pd
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


def materialize_mapping(
    input_dataframe: pd.DataFrame, target: List[dict]
) -> pd.DataFrame:
    output_dataframe = pd.DataFrame()
    for mapping_spec in target:
        from_column_name = mapping_spec["from"]
        to_column_name = mapping_spec["to"]
        value_mapper = mapping_spec["mapper"]
        output_dataframe[to_column_name] = map_column_values(
            input_dataframe[from_column_name], to_column_name, value_mapper
        )
    return output_dataframe


def map_column_values(
    input_column: pd.Series, target: str, value_mapper: ValueMapper
) -> pd.Series:
    new_column = value_mapper.map(input_column)
    new_column.name = target
    return new_column
