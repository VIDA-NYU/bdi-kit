from abc import ABCMeta, abstractmethod
from typing import List
import pandas as pd
import numpy as np


class ColumnEmbedder(metaclass=ABCMeta):
    """
    Base class for column embedding algorithms. Implementations of this class
    must create embeddings for each of the columns of a table.
    """

    @abstractmethod
    def get_embeddings(self, table: pd.DataFrame) -> List[np.ndarray]:
        """
        Must compute a vector embedding for each column in the table.
        The vectors must be represented as dense np.ndarray vectors and
        appear in the same order they appear in the DataFrame `table`.
        """
        pass
