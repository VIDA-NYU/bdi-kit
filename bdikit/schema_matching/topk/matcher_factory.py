from enum import Enum
from typing import Mapping, Any, Type
from bdikit.schema_matching.topk.base import BaseTopkSchemaMatcher
from bdikit.schema_matching.topk import CLTopkSchemaMatcher


class TopkMatchers(Enum):
    CT_LEARNING = ("ct_learning", CLTopkSchemaMatcher)

    def __init__(self, method_name: str, method_class: Type[BaseTopkSchemaMatcher]):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(
        method_name: str, **method_kwargs: Mapping[str, Any]
    ) -> BaseTopkSchemaMatcher:
        methods = {method.method_name: method.method_class for method in TopkMatchers}
        try:
            return methods[method_name](**method_kwargs)
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )
