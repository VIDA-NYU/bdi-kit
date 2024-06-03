from enum import Enum


class ColumnType(Enum):
    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"
    # TODO semantic types?


class Column:
    def __init__(
        self,
        df_name,
        column_name,
        column_type=ColumnType.STRING,
        domain_values=None,
        null_values_representations=None,
    ):
        self.df_name = df_name
        self.column_name = column_name
        self.column_type = column_type

        if domain_values is None:
            self.domain_values = set()
        else:
            self.domain_values = set(domain_values)

        if null_values_representations is None:
            self.null_values_representations = set()
        else:
            self.null_values_representations = set(null_values_representations)

    def __str__(self):
        return f"Column(df_name={self.df_name}, column_name={self.column_name}, column_type={self.column_type}, domain_values={self.domain_values}, null_values_representations={self.null_values_representations})"

    def __eq__(self, value):
        if not isinstance(value, Column):
            return False
        return self.df_name == value.df_name and self.column_name == value.column_name

    def __hash__(self):
        return hash((self.df_name, self.column_name))
