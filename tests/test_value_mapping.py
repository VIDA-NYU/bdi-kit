import pandas as pd
import numpy as np
from bdikit.mapping_algorithms.value_mapping.value_mappers import (
    FunctionValueMapper,
    DictionaryMapper,
    IdentityValueMapper,
)


def test_identity_mapper():
    # given
    str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
    identity_mapper = IdentityValueMapper()

    # when
    mapped_column = identity_mapper.map(str_column)

    # then
    assert mapped_column.eq(["a", "b", "c", "d", "e"]).all()


def test_dictionary_mapper():
    # given
    str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
    dict_mapper = DictionaryMapper(dictionary={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

    # when
    mapped_column = dict_mapper.map(str_column)

    # then
    assert mapped_column.eq([1, 2, 3, 4, 5]).all()


def test_dictionary_mapper_missing_key():
    # given
    str_column = pd.Series(data=["a", "b", "c", "d", "e", None, np.nan], name="column_str")

    # when
    dict_mapper = DictionaryMapper(dictionary={"a": "1", "b": "2", "c": "3", "d": "4"}, missing_key_value=np.nan)
    mapped_column = dict_mapper.map(str_column)
    # then
    assert mapped_column[0:4].tolist() == ['1', '2', '3', '4']
    assert np.isnan(mapped_column[4])
    assert np.isnan(mapped_column[5])
    assert np.isnan(mapped_column[6])

    # when
    dict_mapper = DictionaryMapper(dictionary={"a": 1, "b": 2, "c": 3, "d": 4}, missing_key_value=np.nan)
    mapped_column = dict_mapper.map(str_column)
    # then
    assert mapped_column[0:4].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(mapped_column[4])
    assert np.isnan(mapped_column[5])
    assert np.isnan(mapped_column[6])


def test_custom_function_mapper():
    # given
    str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
    fn_mapper = FunctionValueMapper(function=lambda x: x + x)

    # when
    mapped_column = fn_mapper.map(str_column)

    # then
    assert mapped_column.eq(["aa", "bb", "cc", "dd", "ee"]).all()
