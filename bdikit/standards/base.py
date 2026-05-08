import pandas as pd
from typing import List, Dict

MANDATORY_METADATA_FIELDS = [
    "attribute_description",
    "value_names",
    "value_descriptions",
]


class BaseStandard:
    """
    Base class for all target standards, e.g. GDC.
    """

    def get_attributes(self) -> List[str]:
        """
        Returns a list of all the attributes (strings) of the standard.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_attribute_values(self, attribute_names: List[str]) -> Dict[str, List]:
        """
        Returns a dictionary where the keys are attribute names and the values are lists of possible values for each attribute.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_attribute_metadata(self, attribute_names: List[str]) -> Dict[str, Dict]:
        """
        Returns a dictionary where the keys are attribute names and the values are dictionaries containing metadata for each attribute.

        Each attribute's metadata dictionary will contain these mandatory fields:
        - `attribute_description`: A description of the attribute.
        - `value_names`: A list of possible values for the attribute.
        - `value_descriptions`: A list of descriptions for each value in `value_names`. The lists `value_names` and `value_descriptions` are parallel.

        Other fields, such as `comment`, can also be included. The values for any additional fields must be strings or lists of strings.

        Example:
        {
            'patient_gender': {
                'attribute_description': 'The gender of the patient.',
                'value_names': ['0', '1', '2'],
                'value_descriptions': ['', '', ''],
                'comment': '0 = Female, 1 = Male, 2 = Gender is not known.'
            }
        }
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_dataframe_rep(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame representation of the standard, where each column in the DataFrame is an attribute in the standard and each row is a possible value for that attribute.
        """
        attributes = self.get_attributes()
        attribute_values = self.get_attribute_values(attributes)
        return pd.DataFrame.from_dict(attribute_values, orient="index").transpose()
