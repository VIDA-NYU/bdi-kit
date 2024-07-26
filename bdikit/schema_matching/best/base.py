import pandas as pd
from typing import Dict


class BaseSchemaMatcher:
    def map(self, source: pd.DataFrame, target: pd.DataFrame) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self, dataset: pd.DataFrame, matches: Dict[str, str]
    ) -> Dict[str, str]:
        for column in dataset.columns:
            if column not in matches:
                matches[column] = ""
        return matches
