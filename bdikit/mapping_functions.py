import pandas as pd
import numpy as np
from typing import Any, Callable
from collections import defaultdict
import inspect


class ValueMapper:
    """
    A ValueMapper represents objects that transform the values in a input
    column to the values from a new output column.
    """

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Every concrete ValueMapper should implement this method, which takes a
        pandas Series as input and returns a new pandas Series with transformed
        values.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return repr(self)


class IdentityValueMapper(ValueMapper):
    """
    A column mapper that maps each value in input column into itself.
    """

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Simply copies the values in input_column to the output column.
        """
        return input_column.copy()

    def __repr__(self) -> str:
        return "{'type': 'identity', 'description': 'Maps each value to itself'}"


class FunctionValueMapper(ValueMapper):
    """
    A column mapper that transforms each value in the input column using the
    provided custom function.
    """

    def __init__(self, function: Callable[[pd.Series], pd.Series]):
        self.function = function

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Applies the given function to each value in input_column to generate
        the output column.
        """
        return input_column.map(self.function)

    def __repr__(self) -> str:
        function_name = getattr(self.function, "__name__", repr(self.function))
        try:
            function_code = inspect.getsource(self.function)
        except (OSError, TypeError):
            function_code = None

        if function_code:
            # Compact representation for lambdas or single-line functions
            code_repr = function_code.strip()
            return f"{{'function': '{function_name}', 'code': {repr(code_repr)}}}"
        else:
            return f"{{'function': '{function_name}'}}"


class DictionaryMapper(ValueMapper):
    """
    A column mapper that transforms each value in the input column using the
    values stored in the provided dictionary.
    """

    def __init__(self, dictionary: dict, missing_key_value: Any = np.nan):
        self.mapping_dict = dict(dictionary)
        self.dictionary = defaultdict(lambda: missing_key_value, dictionary)

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Transforms the values in the input_column to the values specified in
        the dictionary provided using the object constructor.
        """
        return input_column.map(self.dictionary, na_action=None)

    def __repr__(self) -> str:
        items = list(self.mapping_dict.items())
        preview = ", ".join(f"{repr(key)}: {repr(value)}" for key, value in items)
        return f"{{{preview}}}"
