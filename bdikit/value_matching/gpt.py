import ast
from typing import List
from openai import OpenAI
from bdikit.value_matching.base import BaseValueMatcher, ValueMatch
from bdikit.config import VALUE_MATCHING_THRESHOLD


class GPT(BaseValueMatcher):
    def __init__(
        self,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        self.client = OpenAI()
        self.threshold = threshold

    def match_values(
        self,
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:

        if (
            len(source_values) > 25
        ):  # TODO: Improve this, avoid calling the API if the number of source values is too high (e.g. IDs)
            return []

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
                        "content": f'For the term: "{source_value}", choose a value from this list {target_values}. '
                        "Return the value from the list with a similarity score, between 0 and 1, with 1 indicating the highest similarity. "
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
                    matches.append(ValueMatch(source_value, target_value, score))
            except:
                print(
                    f'Errors parsing response for "{source_value}": {response_message}'
                )

        return matches
