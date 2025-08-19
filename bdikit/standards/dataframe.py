import pandas as pd
from typing import List, Dict
from bdikit.standards.base import BaseStandard


class DataFrame(BaseStandard):

    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def get_attributes(self) -> List[str]:
        return list(self.dataframe.columns)

    def get_attribute_values(self, attribute_names: List[str]) -> Dict[str, List]:
        attribute_values = {}

        attribute_values = {
            attribute_name: self.dataframe[attribute_name].unique().tolist()
            for attribute_name in attribute_names
        }

        return attribute_values

    def get_attribute_metadata(self, attribute_names: List[str]) -> Dict[str, Dict]:
        attribute_metadata = {}

        for attribute_name in attribute_names:
            attribute_metadata[attribute_name] = {}
            attribute_metadata[attribute_name]["description"] = ""
            attribute_metadata[attribute_name]["value_names"] = (
                self.get_attribute_values([attribute_name])[attribute_name]
            )
            attribute_metadata[attribute_name]["value_descriptions"] = []

        return attribute_metadata

    def get_dataframe_rep(self) -> pd.DataFrame:
        return self.dataframe
