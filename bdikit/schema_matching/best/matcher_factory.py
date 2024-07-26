from enum import Enum
from typing import Mapping, Any, Type
from bdikit.schema_matching.best.base import BaseSchemaMatcher
from bdikit.schema_matching.best import (
    SimFloodSchemaMatcher,
    ComaSchemaMatcher,
    CupidSchemaMatcher,
    DistributionBasedSchemaMatcher,
    JaccardSchemaMatcher,
    GPTSchemaMatcher,
    ContrastiveLearningSchemaMatcher,
    TwoPhaseSchemaMatcher,
    MaxValSimSchemaMatcher,
)


class SchemaMatchers(Enum):
    SIMFLOOD = ("similarity_flooding", SimFloodSchemaMatcher)
    COMA = ("coma", ComaSchemaMatcher)
    CUPID = ("cupid", CupidSchemaMatcher)
    DISTRIBUTION_BASED = ("distribution_based", DistributionBasedSchemaMatcher)
    JACCARD_DISTANCE = ("jaccard_distance", JaccardSchemaMatcher)
    GPT = ("gpt", GPTSchemaMatcher)
    CT_LEARNING = ("ct_learning", ContrastiveLearningSchemaMatcher)
    TWO_PHASE = ("two_phase", TwoPhaseSchemaMatcher)
    MAX_VAL_SIM = ("max_val_sim", MaxValSimSchemaMatcher)

    def __init__(self, method_name: str, method_class: Type[BaseSchemaMatcher]):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(
        method_name: str, **method_kwargs: Mapping[str, Any]
    ) -> BaseSchemaMatcher:
        methods = {method.method_name: method.method_class for method in SchemaMatchers}

        try:
            return methods[method_name](**method_kwargs)
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )
