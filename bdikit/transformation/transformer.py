import json
import pandas as pd


class Transformer:
    def __init__(self, schema_path):
        schema = json.load(open(schema_path))
        if not self.is_schema_valid(schema):
            raise ValueError("Invalid schema")
        self.schema = schema

    def transform(self, df: pd.DataFrame):
        for column in self.schema:
            if column not in df.columns:
                continue
            match = self.schema[column]["match"]
            default = self.schema[column]["default"]
            values = self.schema[column]["values"]
            values = {k: ", ".join(vs) for k, vs in values.items()}
            print(values)

            if self.schema[column].get("ignore_case"):
                df[column] = df[column].str.lower().map(values).fillna(default)
            else:
                df[column] = df[column].map(values).fillna(default)

        return df

    @staticmethod
    def is_schema_valid(schema):
        return True
