from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, TypedDict
import pandas as pd


class ColumnScore(NamedTuple):
    column_name: str
    score: float


class TopkMatching(TypedDict):
    source_column: str
    top_k_columns: List[ColumnScore]


class BaseTopkSchemaMatcher(metaclass=ABCMeta):
    @abstractmethod
    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        pass
