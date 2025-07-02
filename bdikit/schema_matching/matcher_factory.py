from enum import Enum
from typing import Mapping, Any
from bdikit.schema_matching.base import BaseSchemaMatcher, BaseTopkSchemaMatcher
from bdikit.utils import create_matcher


class SchemaMatchers(Enum):
    SIMFLOOD = (
        "similarity_flooding",
        "bdikit.schema_matching.valentine.SimFlood",
    )
    COMA = (
        "coma",
        "bdikit.schema_matching.valentine.Coma",
    )
    CUPID = (
        "cupid",
        "bdikit.schema_matching.valentine.Cupid",
    )
    DISTRIBUTION_BASED = (
        "distribution_based",
        "bdikit.schema_matching.valentine.DistributionBased",
    )
    JACCARD_DISTANCE = (
        "jaccard_distance",
        "bdikit.schema_matching.valentine.Jaccard",
    )

    TWO_PHASE = (
        "two_phase",
        "bdikit.schema_matching.twophase.TwoPhase",
    )

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path


class TopkSchemaMatchers(Enum):
    CT_LEARNING = (
        "ct_learning",
        "bdikit.schema_matching.contrastivelearning.ContrastiveLearning",
    )

    MAX_VAL_SIM = (
        "max_val_sim",
        "bdikit.schema_matching.maxvalsim.MaxValSim",
    )

    MAGNETO_ZS_BP = (
        "magneto_zs_bp",
        "bdikit.schema_matching.magneto.MagnetoZSBP",
    )

    MAGNETO_FT_BP = (
        "magneto_ft_bp",
        "bdikit.schema_matching.magneto.MagnetoFTBP",
    )

    MAGNETO_ZS_LLM = (
        "magneto_zs_llm",
        "bdikit.schema_matching.magneto.MagnetoZSLLM",
    )

    MAGNETO_FT_LLM = (
        "magneto_ft_llm",
        "bdikit.schema_matching.magneto.MagnetoFTLLM",
    )

    LLM = ("llm", "bdikit.schema_matching.llm.LLM")

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path


schema_matchers = {
    method.matcher_name: method.matcher_path for method in SchemaMatchers
}
topk_schema_matchers = {
    method.matcher_name: method.matcher_path for method in TopkSchemaMatchers
}
schema_matchers.update(topk_schema_matchers)


def get_schema_matcher(
    matcher_name: str, **matcher_kwargs: Mapping[str, Any]
) -> BaseSchemaMatcher:

    return create_matcher(matcher_name, schema_matchers, **matcher_kwargs)


def get_topk_schema_matcher(
    matcher_name: str, **matcher_kwargs: Mapping[str, Any]
) -> BaseTopkSchemaMatcher:

    return create_matcher(matcher_name, topk_schema_matchers, **matcher_kwargs)
