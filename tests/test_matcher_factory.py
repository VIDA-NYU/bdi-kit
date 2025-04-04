import os

from bdikit.schema_matching.base import BaseOne2oneSchemaMatcher, BaseTopkSchemaMatcher
from bdikit.schema_matching.matcher_factory import (
    get_one2one_schema_matcher,
    get_topk_schema_matcher,
    One2oneSchemaMatchers,
    TopkSchemaMatchers,
)
from bdikit.value_matching.base import BaseOne2oneValueMatcher, BaseTopkValueMatcher
from bdikit.value_matching.matcher_factory import (
    get_one2one_value_matcher,
    get_topk_value_matcher,
    One2OneValueMatchers,
    TopkValueMatchers,
)


# Set a mock API key to test only the initialization of the matcher
# This is necessary because the OpenAI client requires an API key to be set
os.environ["OPENAI_API_KEY"] = "mock_key"


def test_get_one2one_schema_matcher():
    for matcher in One2oneSchemaMatchers:
        obj = get_one2one_schema_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseOne2oneSchemaMatcher)


def test_get_topk_schema_matcher():
    for matcher in TopkSchemaMatchers:
        obj = get_topk_schema_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseTopkSchemaMatcher)


def test_get_one2one_value_matcher():
    for matcher in One2OneValueMatchers:
        obj = get_one2one_value_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseOne2oneValueMatcher)


def test_get_topk_value_matcher():
    for matcher in TopkValueMatchers:
        obj = get_topk_value_matcher(matcher.matcher_name)
        assert isinstance(obj, BaseTopkValueMatcher)
