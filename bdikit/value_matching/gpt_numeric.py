import re
import random
import textwrap
import numpy as np
from typing import List, Dict
from openai import OpenAI
from bdikit.value_matching.base import BaseValueMatcher, ValueMatch

random.seed(42)


class GPTNumeric(BaseValueMatcher):
    def __init__(
        self,
        sample_size: int = 5,
    ):
        self.client = OpenAI()
        self.sample_size = sample_size

    def sanitize_code(self, function_code):
        function_code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", function_code)
        function_code = re.sub(r"(\s|`)*$", "", function_code)
        function_code = textwrap.dedent(function_code)

        return function_code

    def match_values(
        self,
        source_values: List[str],
        target_values: List[str],
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:

        matches = []
        filtered_values = [x for x in source_values if not np.isnan(x)]
        sample_values = random.sample(filtered_values, self.sample_size)

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent system designed to derive formulas and write Python code "
                    "to convert numeric source values into corresponding target values.",
                },
                {
                    "role": "user",
                    "content": "You are an intelligent system designed to derive formulas and write Python code "
                    "to convert numeric source values into corresponding target values. "
                    f"Given a dataset attribute named '{source_context['attribute_name']}' containing the values: {str(sample_values)}, "
                    f"and a potential target attribute named '{target_context['attribute_name']}' described as: '{target_context['attribute_description']}'. "
                    "Your task is to determine the formula needed to transform source values into target values. "
                    "Write a Python function named 'map_values' that takes a single input value and applies the derived formula to return the corresponding target value. "
                    "Provide only the Python code snippet as a formatted string.",
                },
            ],
        )

        function_code = completion.choices[0].message.content
        function_code = self.sanitize_code(function_code)

        local_scope = {}
        try:
            exec(function_code, {}, local_scope)
        except Exception as e:
            print(f"Error executing function code: {function_code}\n Error: {e}")
            return matches

        map_values_func = local_scope.get("map_values")

        for source_value in source_values:
            try:
                target_value = map_values_func(source_value)
                matches.append(ValueMatch(source_value, target_value, 1.0))
            except Exception as e:
                print(f"Error applying function to value {source_value}: {e}")

        return matches
