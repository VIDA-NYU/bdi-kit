import pandas as pd
import numpy as np
from typing import List
from bdikit.schema_matching.topk.base import (
    ColumnScore,
    TopkMatching,
    BaseTopkSchemaMatcher,
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from bdikit.models.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
    DEFAULT_CL_MODEL,
)
from bdikit.models import ColumnEmbedder


class EmbeddingSimilarityTopkSchemaMatcher(BaseTopkSchemaMatcher):
    def __init__(self, column_embedder: ColumnEmbedder, metric: str = "cosine"):
        self.api = column_embedder
        self.metric = metric

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
        if self.metric == "cosine":
            sim = cosine_similarity(l_features, r_features)  # type: ignore
        elif self.metric == "euclidean":
            sim = euclidean_distances(l_features, r_features)  # type: ignore
            sim = 1 / (1 + sim)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

        top_k_results = []
        for index, similarities in enumerate(sim):
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


class CLTopkSchemaMatcher(EmbeddingSimilarityTopkSchemaMatcher):
    def __init__(self, model_name: str = DEFAULT_CL_MODEL, metric: str = "cosine"):
        super().__init__(
            column_embedder=ContrastiveLearningAPI(model_name=model_name), metric=metric
        )
