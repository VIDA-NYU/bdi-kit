import unittest
import pandas as pd
from bdikit.mapping_algorithms.value_mapping import (
    map_column_values,
    materialize_mapping,
    FunctionValueMapper,
    DictionaryMapper,
    IdentityValueMapper,
)


class ValueMappingTest(unittest.TestCase):

    def test_identity_mapper(self):
        # given
        str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
        identity_mapper = IdentityValueMapper()

        # when
        mapped_column = identity_mapper.map(str_column)

        # then
        self.assertTrue(mapped_column.eq(["a", "b", "c", "d", "e"]).all())

    def test_dictionary_mapper(self):
        # given
        str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
        dict_mapper = DictionaryMapper(
            dictionary={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        )

        # when
        mapped_column = dict_mapper.map(str_column)

        # then
        self.assertTrue(mapped_column.eq([1, 2, 3, 4, 5]).all())

    def test_custom_function_mapper(self):
        # given
        str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
        fn_mapper = FunctionValueMapper(function=lambda x: x + x)

        # when
        mapped_column = fn_mapper.map(str_column)

        # then
        self.assertTrue(mapped_column.eq(["aa", "bb", "cc", "dd", "ee"]).all())

    def test_map_column_values(self):
        """
        Ensures that the map_column_values function correctly maps the values of
        a column and assings the target name.
        """
        # given
        str_column = pd.Series(data=["a", "b", "c", "d", "e"], name="column_str")
        value_mapper = FunctionValueMapper(function=lambda x: x.upper())
        target_column_name = "string column"

        # when
        mapped_column = map_column_values(
            str_column, target=target_column_name, value_mapper=value_mapper
        )

        # then
        upper_cased_values = ["A", "B", "C", "D", "E"]
        self.assertTrue(mapped_column.name == target_column_name)
        self.assertTrue(mapped_column.eq(upper_cased_values).all())

    def test_map_dataframe_column_values(self):
        # given
        str_column_1 = ["a", "b", "c", "d", "e"]
        str_column_2 = ["a", "b", "c", "d", "e"]
        df_base = pd.DataFrame(
            {"column_str_1": str_column_1, "column_str_2": str_column_2}
        )

        value_mapping_spec = [
            {
                "from": "column_str_1",
                "to": "string column 1",
                "mapper": IdentityValueMapper(),
            },
            {
                "from": "column_str_2",
                "to": "string column 2",
                "mapper": FunctionValueMapper(function=lambda x: x.upper()),
            },
        ]

        # when
        df_mapped = materialize_mapping(df_base, target=value_mapping_spec)

        # then
        self.assertTrue(len(df_mapped.columns) == 2)

        self.assertTrue("string column 1" in df_mapped.columns)
        self.assertTrue(df_mapped["string column 1"].eq(str_column_1).all())

        self.assertTrue("string column 2" in df_mapped.columns)
        self.assertTrue(
            df_mapped["string column 2"].eq(["A", "B", "C", "D", "E"]).all()
        )
