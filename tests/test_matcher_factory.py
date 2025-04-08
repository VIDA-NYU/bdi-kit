import os

from bdikit.schema_matching.base import BaseSchemaMatcher, BaseTopkSchemaMatcher
from bdikit.schema_matching.matcher_factory import (
    get_schema_matcher,
    get_topk_schema_matcher,
    SchemaMatchers,
    TopkSchemaMatchers,
)
from bdikit.value_matching.base import BaseValueMatcher, BaseTopkValueMatcher
from bdikit.value_matching.matcher_factory import (
    get_value_matcher,
    get_topk_value_matcher,
    ValueMatchers,
    TopkValueMatchers,
)


# Set a mock API key to test only the initialization of the matcher
# This is necessary because the OpenAI client requires an API key to be set
os.environ["OPENAI_API_KEY"] = "mock_key"


def test_get_schema_matcher():
    for matcher in SchemaMatchers:
        obj = get_schema_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseSchemaMatcher)


def test_get_topk_schema_matcher():
    for matcher in TopkSchemaMatchers:
        obj = get_topk_schema_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseTopkSchemaMatcher)


def test_get_value_matcher():
    for matcher in ValueMatchers:
        obj = get_value_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseValueMatcher)


def test_get_topk_value_matcher():
    for matcher in TopkValueMatchers:
        obj = get_topk_value_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseTopkValueMatcher)
