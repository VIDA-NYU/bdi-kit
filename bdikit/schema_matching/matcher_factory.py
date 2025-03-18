import importlib
from enum import Enum
from typing import Mapping, Dict, Any
from bdikit.schema_matching.base import BaseOne2oneSchemaMatcher, BaseTopkSchemaMatcher


class One2oneSchemaMatchers(Enum):
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
    GPT = ("gpt", "bdikit.schema_matching.gpt.GPT")

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
        "bdikit.schema_matching.topk.maxvalsim.MaxValSim",
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

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path


one2one_schema_matchers = {
    method.matcher_name: method.matcher_path for method in One2oneSchemaMatchers
}
topk_schema_matchers = {
    method.matcher_name: method.matcher_path for method in TopkSchemaMatchers
}
one2one_schema_matchers.update(topk_schema_matchers)


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
    # Load the class dynamically
    module_path, class_name = available_matchers[matcher_name].rsplit(".", 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)(**matcher_kwargs)


def get_one2one_schema_matcher(
    matcher_name: str, **matcher_kwargs: Mapping[str, Any]
) -> BaseOne2oneSchemaMatcher:

    return create_matcher(matcher_name, one2one_schema_matchers, **matcher_kwargs)


def get_topk_schema_matcher(
    matcher_name: str, **matcher_kwargs: Mapping[str, Any]
) -> BaseTopkSchemaMatcher:

    return create_matcher(matcher_name, topk_schema_matchers, **matcher_kwargs)
