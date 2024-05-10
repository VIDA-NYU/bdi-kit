from enum import Enum

class ColumnType(Enum):
        STRING = 'string'
        FLOAT = 'float'
        INTEGER = 'integer'
        UNDEFINED = 'undefined'

class Column:
    def __init__(self, column_name, column_type=None, unique_values=None, unique_null_values=None, column_group=None):
        self.name = column_name
        self.type = column_type
        self.type = column_type
        self.group = column_group
        self.unique_values = unique_values
        self.unique_null_values = unique_null_values
            
    def __str__(self):
        return f"Column({self.column_name}:{self.column_type})"

class Table:
    def __init__(self, table_name, columns):
        self.name = table_name
        self.columns = columns

    @classmethod
    def from_dataframe(cls, df):
        columns = []
        for col in df.columns:
            unique_values = df[col].unique()
            #unique_null_values = df[col].isnull().sum()
            column_type = ColumnType.STRING
            columns.append(Column(col, column_type, unique_values))
        return cls(df.name, columns)

    def __str__(self):
        return f"Table({self.table_name})"
    


