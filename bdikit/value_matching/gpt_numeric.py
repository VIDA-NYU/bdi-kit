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

    def match_values(
        self,
        source_values: List[str],
        target_values: List[str],
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:

        matches = []
        sample_values = []
        # For cases where source values are numeric but in string format, e.g. "1.0", "2.0"
        clean_source_values = []
        for source_value in source_values:
            formatted_value = to_number(source_value)
            if formatted_value is not None and len(sample_values) < self.sample_size:
                sample_values.append(formatted_value)
            clean_source_values.append(formatted_value)

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
        function_code = sanitize_code(function_code)
        print(f"Function code: {function_code}")
        local_scope = {}
        try:
            exec(function_code, {}, local_scope)
        except Exception as e:
            print(f"Error executing function code: {function_code}\n Error: {e}")
            return matches

        map_values_func = local_scope.get("map_values")

        for index, clean_source_value in enumerate(clean_source_values):
            try:
                if clean_source_value is None:
                    continue
                target_value = map_values_func(clean_source_value)
                matches.append(ValueMatch(source_values[index], target_value, 1.0))
            except Exception as e:
                print(
                    f"Error applying function to value '{source_value}' {type(source_value)}': {e}"
                )

        return matches


def sanitize_code(function_code):
    function_code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", function_code)
    function_code = re.sub(r"(\s|`)*$", "", function_code)
    function_code = textwrap.dedent(function_code)

    return function_code


def to_number(value):
    try:
        # Attempt to convert to float
        num = float(value)
        # Check if it is NaN
        if np.isnan(num):
            return None
        # If the float is an integer, return it as an int
        return int(num) if num.is_integer() else num
    except (ValueError, TypeError):
        # Return None if conversion fails
        return None
