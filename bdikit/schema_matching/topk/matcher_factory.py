import importlib
from enum import Enum
from typing import Mapping, Any
from bdikit.schema_matching.topk.base import BaseTopkSchemaMatcher


class TopkMatchers(Enum):
    CT_LEARNING = (
        "ct_learning",
        "bdikit.schema_matching.topk.contrastivelearning.CLTopkSchemaMatcher",
    )

    MAGNETO_ZS_BP = (
        "magneto_zs_bp",
        "bdikit.schema_matching.topk.magneto.MagnetoZSBP",
    )

    MAGNETO_FT_BP = (
        "magneto_ft_bp",
        "bdikit.schema_matching.topk.magneto.MagnetoFTBP",
    )

    MAGNETO_ZS_LLM = (
        "magneto_zs_llm",
        "bdikit.schema_matching.topk.magneto.MagnetoZSLLM",
    )

    MAGNETO_FT_LLM = (
        "magneto_ft_llm",
        "bdikit.schema_matching.topk.magneto.MagnetoFTLLM",
    )

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path

    @staticmethod
    def get_matcher(
        matcher_name: str, **matcher_kwargs: Mapping[str, Any]
    ) -> BaseTopkSchemaMatcher:
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


matchers = {method.matcher_name: method.matcher_path for method in TopkMatchers}
