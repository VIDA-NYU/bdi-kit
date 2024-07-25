from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, TypedDict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bdikit.models.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
    DEFAULT_CL_MODEL,
)
from bdikit.models import ColumnEmbedder
from bdikit.models.splade import SpladeEmbedder


class ColumnScore(NamedTuple):
    column_name: str
    score: float


class TopkMatching(TypedDict):
    source_column: str
    top_k_columns: List[ColumnScore]


class TopkColumnMatcher(metaclass=ABCMeta):
    @abstractmethod
    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        pass


class EmbeddingSimilarityTopkColumnMatcher(TopkColumnMatcher):
    def __init__(self, column_embedder: ColumnEmbedder):
        self.api = column_embedder

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


class CLTopkColumnMatcher(EmbeddingSimilarityTopkColumnMatcher):
    def __init__(self, model_name: str = DEFAULT_CL_MODEL):
        super().__init__(column_embedder=ContrastiveLearningAPI(model_name=model_name))


class SpladeTopkColumnMatcher(EmbeddingSimilarityTopkColumnMatcher):
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        super().__init__(column_embedder=SpladeEmbedder(model_id=model_name))


class SpladeMaxSimTopkColumnMatcher(CLTopkColumnMatcher):
    def __init__(
        self,
    ):
        super().__init__()
        self.splade = SpladeEmbedder(model_id="naver/splade-cocondenser-ensembledistil")

    def unique_string_values(self, column: pd.Series) -> pd.Series:
        if pd.api.types.is_string_dtype(column):
            return pd.Series(column.unique(), name=column.name)
        else:
            return pd.Series(column.unique().astype(str), name=column.name)

    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int = 10
    ) -> List[TopkMatching]:
        """
        Returns the top-k matching columns in the target table for each column
        in the source table. The ranking is based on the cosine similarity of
        the embeddings of the columns in the source and target tables.
        """
        top_k_cl = super().get_recommendations(source, target, top_k)
        top_k_results = []
        for column_topk in top_k_cl:
            # rerank top_k_columns based on ColumnSimilarity scores
            new_scores: List[ColumnScore] = []
            source_column = self.unique_string_values(
                source[column_topk["source_column"]]
            )
            for target_score in column_topk["top_k_columns"]:
                target_column = self.unique_string_values(
                    target[target_score.column_name]
                )
                score = self.similarity(source_column, target_column)
                new_scores.append(
                    ColumnScore(column_name=target_score.column_name, score=score)
                )

            new_scores = sorted(new_scores, key=lambda x: x.score, reverse=True)
            top_k_results.append(
                {
                    "source_column": column_topk["source_column"],
                    "top_k_columns": new_scores[:top_k],
                }
            )

        return top_k_results

    def similarity(self, source: pd.Series, target: pd.Series) -> float:
        assert isinstance(
            source.name, str
        ), f"Column header must be a string but was: {source.name}"
        assert isinstance(
            target.name, str
        ), f"Column header must be a string but was: {target.name}"
        source_header_emb = self.splade.embed_values([source.name]).column
        target_header_emb = self.splade.embed_values([target.name]).column
        header_sim = cosine_similarity([source_header_emb], [target_header_emb])[0][0]

        source_embeddings = self.splade.embed_values(
            source.to_list(), embed_values=True
        )
        target_embeddings = self.splade.embed_values(
            target.to_list(), embed_values=True
        )

        max_similarities = []
        for _, source_value_vec in source_embeddings.values.items():
            max_sim = 0
            for _, target_value_vec in target_embeddings.values.items():
                sim = cosine_similarity([source_value_vec], [target_value_vec])[0][0]
                if sim > max_sim:
                    max_sim = sim
            max_similarities.append(sim)
        values_sim = np.mean(max_similarities)

        return 0.5 * header_sim + 0.5 * values_sim
