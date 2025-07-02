from enum import Enum
from typing import Mapping, Any
from bdikit.value_matching.base import BaseValueMatcher, BaseTopkValueMatcher
from bdikit.utils import create_matcher


class ValueMatchers(Enum):
    EDIT = (
        "edit_distance",
        "bdikit.value_matching.polyfuzz.EditDistance",
    )
    LLM = ("llm", "bdikit.value_matching.llm.LLM")
    LLM_NUMERIC = ("llm_numeric", "bdikit.value_matching.llm_numeric.LLMNumeric")

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path


class TopkValueMatchers(Enum):
    TFIDF = ("tfidf", "bdikit.value_matching.polyfuzz.TFIDF")

    EMBEDDINGS = (
        "embedding",
        "bdikit.value_matching.polyfuzz.Embedding",
    )
    FASTTEXT = (
        "fasttext",
        "bdikit.value_matching.polyfuzz.FastText",
    )

    def __init__(self, matcher_name: str, matcher_path: str):
        self.matcher_name = matcher_name
        self.matcher_path = matcher_path


value_matchers = {method.matcher_name: method.matcher_path for method in ValueMatchers}
topk_value_matchers = {
    method.matcher_name: method.matcher_path for method in TopkValueMatchers
}
value_matchers.update(topk_value_matchers)


def get_value_matcher(
    matcher_name: str, **matcher_kwargs: Mapping[str, Any]
) -> BaseValueMatcher:

    return create_matcher(matcher_name, value_matchers, **matcher_kwargs)


def get_topk_value_matcher(
    matcher_name: str, **matcher_kwargs: Mapping[str, Any]
) -> BaseTopkValueMatcher:

    return create_matcher(matcher_name, topk_value_matchers, **matcher_kwargs)
