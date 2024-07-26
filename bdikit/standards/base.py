import pandas as pd
from typing import List, Dict


class BaseStandard:
    """
    Base class for all target standards, e.g. GDC.
    """

    def get_columns(self) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_column_values(self, column_names: List[str]) -> Dict[str, List]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_column_metadata(self, column_names: List[str]) -> Dict[str, Dict]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_dataframe_rep(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this method")
