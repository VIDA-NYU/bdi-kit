import importlib
from enum import Enum
from typing import Mapping, Any
from bdikit.schema_matching.best.base import BaseSchemaMatcher


class SchemaMatchers(Enum):
    SIMFLOOD = (
        "similarity_flooding",
        "bdikit.schema_matching.best.valentine.SimFloodSchemaMatcher",
    )
    COMA = (
        "coma",
        "bdikit.schema_matching.best.valentine.ComaSchemaMatcher",
    )
    CUPID = (
        "cupid",
        "bdikit.schema_matching.best.valentine.CupidSchemaMatcher",
    )
    DISTRIBUTION_BASED = (
        "distribution_based",
        "bdikit.schema_matching.best.valentine.DistributionBasedSchemaMatcher",
    )
    JACCARD_DISTANCE = (
        "jaccard_distance",
        "bdikit.schema_matching.best.valentine.JaccardDistanceSchemaMatcher",
    )
    GPT = ("gpt", "bdikit.schema_matching.best.gpt.GPTSchemaMatcher")
    CT_LEARNING = (
        "ct_learning",
        "bdikit.schema_matching.best.contrastivelearning.ContrastiveLearningSchemaMatcher",
    )
    TWO_PHASE = (
        "two_phase",
        "bdikit.schema_matching.best.twophase.TwoPhaseSchemaMatcher",
    )
    MAX_VAL_SIM = (
        "max_val_sim",
        "bdikit.schema_matching.best.maxvalsim.MaxValSimSchemaMatcher",
    )

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path

    @staticmethod
    def get_matcher(
        matcher_name: str, **matcher_kwargs: Mapping[str, Any]
    ) -> BaseSchemaMatcher:
        if matcher_name not in matchers:
            names = ", ".join(list(matchers.keys()))
            raise ValueError(
                f"The {matcher_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )
        # Load the class dynamically
        module_path, class_name = matchers[matcher_name].rsplit(".", 1)
        module = importlib.import_module(module_path)

        return getattr(module, class_name)(**matcher_kwargs)


matchers = {method.matcher_name: method.matcher_path for method in SchemaMatchers}
