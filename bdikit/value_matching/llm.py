import ast
from typing import List, Dict, Any
from openai import OpenAI
from bdikit.value_matching.base import BaseValueMatcher, ValueMatch
from bdikit.utils import get_additional_context
from bdikit.config import VALUE_MATCHING_THRESHOLD


class LLM(BaseValueMatcher):
    """A value matcher that uses LLM to match values based on their similarity."""

    def __init__(
        self,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        self.client = OpenAI()
        self.threshold = threshold

    def match_values(
        self,
        source_values: List[Any],
        target_values: List[Any],
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:

        if (
            len(source_values) > 25
        ):  # TODO: Improve this, avoid calling the API if the number of source values is too high (e.g. IDs)
            return []
        source_attribute = source_context["attribute_name"]
        target_attribute = target_context["attribute_name"]
        additional_source_cxt = get_additional_context(source_context, "source")
        additional_target_cxt = get_additional_context(target_context, "target")
        additional_context = additional_source_cxt + additional_target_cxt

        target_values_set = set(target_values)
        matches = []

        for source_value in source_values:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent system that given a term, you have to choose a value from a list that best matches the term.",
                    },
                    {
                        "role": "user",
                        "content": f"For the term: '{source_value}', choose a value from this list {target_values}. "
                        "Return the value from the list with a similarity score, between 0 and 1, with 1 indicating the highest similarity. "
                        f"{additional_context}"
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
                        'Only provide a Python dictionary. For example {"term": "term from the list", "score": 0.8}.',
                    },
                ],
            )

            response_message = completion.choices[0].message.content
            try:
                response_dict = ast.literal_eval(response_message)
                target_value = response_dict["term"]
                score = float(response_dict["score"])
                if target_value in target_values_set and score >= self.threshold:
                    matches.append(
                        ValueMatch(
                            source_attribute,
                            target_attribute,
                            source_value,
                            target_value,
                            score,
                        )
                    )
            except:
                print(
                    f'Errors parsing response for "{source_value}": {response_message}'
                )

        matches = self._sort_matches(matches)

        return self._fill_missing_matches(
            source_values, matches, source_attribute, target_attribute
        )
