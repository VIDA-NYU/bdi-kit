import pandas as pd
import numpy as np
from typing import Any, Callable
from collections import defaultdict


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


class IdentityValueMapper(ValueMapper):
    """
    A column mapper that maps each value in input column into itself.
    """

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Simply copies the values in input_column to the output column.
        """
        return input_column.copy()


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


class DictionaryMapper(ValueMapper):
    """
    A column mapper that transforms each value in the input column using the
    values stored in the provided dictionary.
    """

    def __init__(self, dictionary: dict, missing_key_value: Any = np.nan):
        self.dictionary = defaultdict(lambda: missing_key_value, dictionary)

    def map(self, input_column: pd.Series) -> pd.Series:
        """
        Transforms the values in the input_column to the values specified in
        the dictionary provided using the object constructor.
        """
        return input_column.map(self.dictionary, na_action=None)
