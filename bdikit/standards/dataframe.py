import pandas as pd
from typing import List, Dict
from bdikit.standards.base import BaseStandard


class DataFrame(BaseStandard):

    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def get_columns(self) -> List[str]:
        return list(self.dataframe.columns)

    def get_column_values(self, column_names: List[str]) -> Dict[str, List]:
        column_values = {}

        column_values = {
            column_name: self.dataframe[column_name].unique().tolist()
            for column_name in column_names
        }

        return column_values

    def get_column_metadata(self, column_names: List[str]) -> Dict[str, Dict]:
        column_metadata = {}

        for column_name in column_names:
            column_metadata[column_name] = {}
            column_metadata[column_name]["description"] = ""
            column_metadata[column_name]["value_names"] = self.get_column_values(
                [column_name]
            )[column_name]
            column_metadata[column_name]["value_descriptions"] = []

        return column_metadata

    def get_dataframe_rep(self) -> pd.DataFrame:
        return self.dataframe
