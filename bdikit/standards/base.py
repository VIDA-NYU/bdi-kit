import pandas as pd
from typing import List, Dict


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
        Returns a dictionary where the keys are attribute names and the values are dictionaries containing these fields for each attribute:
        `description`, `value_names`, and `value_descriptions`.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_dataframe_rep(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame representation of the standard, where each column in the DataFrame is an attribute in the standard and each row is a possible value for that attribute.
        """
        raise NotImplementedError("Subclasses must implement this method")
