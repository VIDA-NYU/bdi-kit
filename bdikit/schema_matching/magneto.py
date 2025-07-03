import pandas as pd
from typing import Dict, Any, List
from magneto import Magneto as Magneto_Lib
from bdikit.download import get_cached_model_or_download
from bdikit.schema_matching.base import ColumnMatch, BaseTopkSchemaMatcher

DEFAULT_MAGNETO_MODEL = "magneto-gdc-v0.1"


class MagnetoBase(BaseTopkSchemaMatcher):
    def __init__(self, kwargs: Dict[str, Any] = None):
        if kwargs is None:
            kwargs = {}
        self.magneto = Magneto_Lib(**kwargs)

    def match_schema(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> List[ColumnMatch]:
        # Temporary workaround due to Magneto's top-1 matching issue
        # Issue details: https://github.com/VIDA-NYU/magneto-matcher/issues/10
        # Once resolved, this function will be removed to use the default implementation in the parent class.
        matches = self.rank_schema_matches(
            source, target, 2
        )  # Get top-2 matches to avoid the issue

        best_matches = {}
        for match in matches:
            if (
                match.source_column not in best_matches
                or match.similarity > best_matches[match.source_column].similarity
            ):
                best_matches[match.source_column] = match

        matches = self._sort_matches(list(best_matches.values()))

        return self._fill_missing_matches(source, matches)

    def rank_schema_matches(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[ColumnMatch]:
        self.magneto.params["topk"] = (
            top_k  # Magneto does not provide a method to set topk
        )
        raw_matches = self.magneto.get_matches(source, target)
        target_columns = set(target.columns)
        matches = []

        for (source_column, target_column), score in raw_matches.items():
            source_column = source_column[1]
            target_column = target_column[1]
            if target_column not in target_columns:
                continue
            matches.append(ColumnMatch(source_column, target_column, score))

        matches = self._sort_ranked_matches(matches)

        return self._fill_missing_matches(source, matches)


class MagnetoZSBP(MagnetoBase):
    """Uses a zero-shot small language model as retriever with the bipartite algorithm as reranker in Magneto."""

    def __init__(self):
        super().__init__()


class MagnetoFTBP(MagnetoBase):
    """Uses a fine-tuned small language model as retriever with the bipartite algorithm as reranker in Magneto."""

    def __init__(
        self,
        encoding_mode: str = "header_values_verbose",
        model_name: str = DEFAULT_MAGNETO_MODEL,
        model_path: str = None,
    ):
        embedding_model = check_magneto_model(model_name, model_path)
        kwargs = {"encoding_mode": encoding_mode, "embedding_model": embedding_model}
        super().__init__(kwargs)


class MagnetoZSLLM(MagnetoBase):
    """Uses a zero-shot small language model as retriever with a large language model as reranker in Magneto."""

    def __init__(self):
        kwargs = {"use_bp_reranker": False, "use_gpt_reranker": True}
        super().__init__(kwargs)


class MagnetoFTLLM(MagnetoBase):
    """Uses a fine-tuned small language model as retriever with a large language model as reranker in Magneto."""

    def __init__(
        self,
        encoding_mode: str = "header_values_verbose",
        model_name: str = DEFAULT_MAGNETO_MODEL,
        model_path: str = None,
    ):
        embedding_model = check_magneto_model(model_name, model_path)
        kwargs = {
            "encoding_mode": encoding_mode,
            "embedding_model": embedding_model,
            "use_bp_reranker": False,
            "use_gpt_reranker": True,
        }
        super().__init__(kwargs)


def check_magneto_model(model_name: str, model_path: str):
    if model_name and model_path:
        raise ValueError(
            "Only one of model_name or model_path should be provided "
            "(they are mutually exclusive)"
        )

    if model_path:
        return model_path
    elif model_name:
        return get_cached_model_or_download(model_name)
    else:
        raise ValueError("Either model_name or model_path must be provided")
