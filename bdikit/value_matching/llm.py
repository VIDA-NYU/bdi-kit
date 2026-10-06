import json_repair
import warnings
from litellm import completion
from typing import List, Dict, Any, Mapping
from bdikit.value_matching.base import BaseTopkValueMatcher, ValueMatch
from bdikit.utils import generate_string_context
from bdikit.config import VALUE_MATCHING_THRESHOLD


class LLM(BaseTopkValueMatcher):
    """A value matcher that uses LLM to match values based on their similarity."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        threshold: float = VALUE_MATCHING_THRESHOLD,
        **model_kwargs: Mapping[str, Any],
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.model_kwargs = model_kwargs

    def rank_value_matches(
        self,
        source_values: List[Any],
        target_values: List[Any],
        top_k: int,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:

        source_attribute = source_context["attribute_name"]
        target_attribute = target_context["attribute_name"]

        additional_source_cxt = generate_string_context(source_context, "source")
        additional_target_cxt = generate_string_context(target_context, "target")
        additional_context = additional_source_cxt + additional_target_cxt

        target_values_set = set(target_values)
        matches = []

        for source_value in source_values:
            response = completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent system that given a term, ranks the values from a list that best match the term.",
                    },
                    {
                        "role": "user",
                        "content": f"For the term: '{source_value}', choose up to {top_k} values from this list {target_values} "
                        "that best match it, ordered from most to least similar. "
                        "Return each chosen value together with a similarity score, between 0 and 1, with 1 indicating the highest similarity. "
                        f"{additional_context}"
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
                        'Only provide a Python list of dictionaries. For example [{"term": "term from the list", "score": 0.8}, '
                        '{"term": "other term from the list", "score": 0.4}].',
                    },
                ],
                **self.model_kwargs,
            )

            response_message = response.choices[0].message.content

            try:
                response_list = json_repair.loads(response_message)
                ranked_matches = []
                for entry in response_list:
                    target_value = entry["term"]
                    score = float(entry["score"])
                    if target_value in target_values_set and score >= self.threshold:
                        ranked_matches.append((target_value, score))

                ranked_matches = sorted(
                    ranked_matches, key=lambda match: match[1], reverse=True
                )[:top_k]

                for target_value, score in ranked_matches:
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
                warnings.warn(
                    f'Errors parsing response for "{source_value}": {response_message}',
                    UserWarning,
                )

        matches = self._sort_matches(matches)

        return self._fill_missing_matches(
            source_values, matches, source_attribute, target_attribute
        )
