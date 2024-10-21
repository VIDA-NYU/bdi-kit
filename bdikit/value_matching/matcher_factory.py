from enum import Enum
from typing import Mapping, Any, Type
from bdikit.value_matching.base import BaseValueMatcher
from bdikit.value_matching import (
    GPTValueMatcher,
    TFIDFValueMatcher,
    EditDistanceValueMatcher,
    EmbeddingValueMatcher,
    FastTextValueMatcher,
)


class ValueMatchers(Enum):
    TFIDF = ("tfidf", TFIDFValueMatcher)
    EDIT = ("edit_distance", EditDistanceValueMatcher)
    EMBEDDINGS = ("embedding", EmbeddingValueMatcher)
    FASTTEXT = ("fasttext", FastTextValueMatcher)
    GPT = ("gpt", GPTValueMatcher)

    def __init__(self, method_name: str, method_class: Type[BaseValueMatcher]):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_instance(
        method_name: str, **method_kwargs: Mapping[str, Any]
    ) -> BaseValueMatcher:
        methods = {method.method_name: method.method_class for method in ValueMatchers}
        try:
            return methods[method_name](**method_kwargs)
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )
