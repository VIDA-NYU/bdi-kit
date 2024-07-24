from typing import List, NamedTuple, TypedDict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bdikit.models.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
    DEFAULT_CL_MODEL,
)


class ColumnScore(NamedTuple):
    column_name: str
    score: float


class TopkMatching(TypedDict):
    source_column: str
    top_k_columns: List[ColumnScore]


class TopkColumnMatcher:
    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:  # type: ignore
        pass


class CLTopkColumnMatcher(TopkColumnMatcher):
    def __init__(self, model_name: str = DEFAULT_CL_MODEL):
        # TODO: we can generalize this api to accept any embedding model
        # and not just our contrastive learning model
        self.api = ContrastiveLearningAPI(model_name=model_name)

    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int = 10
    ) -> List[TopkMatching]:
        """
        Returns the top-k matching columns in the target table for each column
        in the source table. The ranking is based on the cosine similarity of
        the embeddings of the columns in the source and target tables.
        """
        l_features = self.api.get_embeddings(source)
        r_features = self.api.get_embeddings(target)
        cosine_sim = cosine_similarity(l_features, r_features)  # type: ignore

        top_k_results = []
        for index, similarities in enumerate(cosine_sim):
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_columns = [
                ColumnScore(column_name=target.columns[i], score=similarities[i])
                for i in top_k_indices
            ]
            top_k_results.append(
                {
                    "source_column": source.columns[index],
                    "top_k_columns": top_k_columns,
                }
            )

        return top_k_results
