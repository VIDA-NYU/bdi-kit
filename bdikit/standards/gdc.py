import json
import pandas as pd
from os.path import join, dirname
from typing import List, Dict
from bdikit.standards.base import BaseStandard


GDC_SCHEMA_PATH = join(dirname(__file__), "../resource/gdc_schema.json")


class GDC(BaseStandard):
    """
    Class for GDC standard.
    """

    def __init__(self) -> None:
        self.data = None
        self.__read_data()

    def __read_data(self):
        with open(GDC_SCHEMA_PATH) as json_file:
            self.data = json.load(json_file)

    def get_attributes(self) -> List[str]:
        return list(self.data.keys())

    def get_attribute_values(
        self, attribute_names: List[str]
    ) -> Dict[str, List]:  # get_gdc_data
        attribute_values = {}

        for attribute_name in attribute_names:
            raw_metadata = self.data.get(attribute_name, {})
            attribute_values[attribute_name] = list(
                raw_metadata.get("value_data", {}).keys()
            )

        return attribute_values

    def get_attribute_metadata(
        self, attribute_names: List[str]
    ) -> Dict[str, Dict]:  # get_gdc_metadata
        attribute_metadata = {}

        for attribute_name in attribute_names:
            raw_metadata = self.data.get(attribute_name, {})
            attribute_metadata[attribute_name] = {}
            attribute_metadata[attribute_name]["description"] = raw_metadata.get(
                "column_description", ""
            )
            attribute_metadata[attribute_name]["value_names"] = list(
                raw_metadata.get("value_data", {}).keys()
            )
            attribute_metadata[attribute_name]["value_descriptions"] = list(
                raw_metadata.get("value_data", {}).values()
            )

        return attribute_metadata

    def get_dataframe_rep(self) -> pd.DataFrame:
        reshaped_data = {
            key: list(value["value_data"].keys()) for key, value in self.data.items()
        }

        # Ensure all lists have the same length by padding with None
        max_length = max(len(v) for v in reshaped_data.values())
        for k, v in reshaped_data.items():
            reshaped_data[k].extend([None] * (max_length - len(v)))

        df = pd.DataFrame.from_dict(reshaped_data, orient="columns")

        return df
