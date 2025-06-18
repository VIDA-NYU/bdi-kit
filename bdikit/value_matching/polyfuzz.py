import flair
import torch
import pandas as pd
from rapidfuzz import fuzz
from polyfuzz import PolyFuzz as PolyFuzzLib
from typing import List, Callable, Tuple, Dict, Any
from bdikit.value_matching.base import (
    BaseTopkValueMatcher,
    BaseValueMatcher,
    ValueMatch,
)
from polyfuzz.models import (
    EditDistance as EditDistanceMatcher,
    TFIDF as TFIDFMatcher,
    Embeddings as EmbeddingMatcher,
)
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings
from bdikit.config import get_device, VALUE_MATCHING_THRESHOLD


flair.device = torch.device(get_device())


class PolyFuzz(BaseTopkValueMatcher):
    """
    Base class for value matching algorithms based on the PolyFuzz library.
    """

    def __init__(self, polyfuzz_model: PolyFuzzLib, threshold: float):
        self.model = polyfuzz_model
        self.threshold = threshold

    def rank_value_matches(
        self,
        source_values: List[Any],
        target_values: List[Any],
        top_k: int,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:

        new_target_values = remove_non_string_values(target_values)
        if len(new_target_values) == 0:
            return []

        new_source_values = remove_non_string_values(source_values)
        if len(new_source_values) == 0:
            return []

        source_attribute = source_context["attribute_name"]
        target_attribute = target_context["attribute_name"]
        self.model.method.top_n = top_k
        self.model.match(new_source_values, new_target_values)
        match_results = self.model.get_matches()

        matches = []
        for _, row in match_results.iterrows():
            source = row[0]
            top_matches = row[1:]
            indexes = range(0, len(top_matches) - 1, 2)

            for index in indexes:
                target = top_matches[index]
                similarity = top_matches[index + 1]
                if similarity >= self.threshold:
                    matches.append(
                        ValueMatch(
                            source_attribute,
                            target_attribute,
                            source,
                            target,
                            similarity,
                        )
                    )

        matches = self._sort_matches(matches)

        return self._fill_missing_matches(
            source_values, matches, source_attribute, target_attribute
        )


class TFIDF(PolyFuzz):
    """
    Value matching algorithm based on the TF-IDF similarity between values.
    """

    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 3),
        clean_string: bool = True,
        threshold: float = VALUE_MATCHING_THRESHOLD,
        cosine_method: str = "sparse",
    ):

        super().__init__(
            PolyFuzzLib(
                method=TFIDFMatcher(
                    n_gram_range=n_gram_range,
                    clean_string=clean_string,
                    min_similarity=threshold,
                    cosine_method=cosine_method,
                )
            ),
            threshold,
        )


class Embedding(PolyFuzz):
    """
    Value matching algorithm based on the cosine similarity of value embeddings.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        threshold: float = VALUE_MATCHING_THRESHOLD,
        cosine_method: str = "sparse",
    ):
        embeddings = TransformerWordEmbeddings(model_name)
        method = EmbeddingMatcher(
            embeddings,
            min_similarity=threshold,
            cosine_method=cosine_method,
        )
        super().__init__(PolyFuzzLib(method), threshold)


class FastText(PolyFuzz):
    """
    Value matching algorithm based on the cosine similarity of FastText embeddings.
    """

    def __init__(
        self,
        model_name: str = "en-crawl",
        threshold: float = VALUE_MATCHING_THRESHOLD,
        cosine_method: str = "sparse",
    ):
        embeddings = WordEmbeddings(model_name)
        method = EmbeddingMatcher(
            embeddings,
            min_similarity=threshold,
            cosine_method=cosine_method,
        )
        super().__init__(PolyFuzzLib(method), threshold)


class EditDistance(BaseValueMatcher):
    """
    Value matching algorithm based on the edit distance between values.
    """

    def __init__(
        self,
        scorer: Callable[[str, str], float] = fuzz.ratio,
        n_jobs: int = -1,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        # Return scores between 0 and 1
        normalized_scorer = lambda str1, str2: scorer(str1, str2) / 100.0

        self.model = PolyFuzzLib(
            method=EditDistanceMatcher(
                n_jobs=n_jobs, scorer=normalized_scorer, normalize=False
            )
        )
        self.threshold = threshold

    def match_values(
        self,
        source_values: List[Any],
        target_values: List[Any],
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:

        new_target_values = remove_non_string_values(target_values)
        if len(new_target_values) == 0:
            return []

        new_source_values = remove_non_string_values(source_values)
        if len(new_source_values) == 0:
            return []
        source_attribute = source_context["attribute_name"]
        target_attribute = target_context["attribute_name"]
        self.model.match(new_source_values, new_target_values)
        match_results = self.model.get_matches()

        matches = []
        for _, row in match_results.iterrows():
            source_value = row["From"]
            target_value = row["To"]
            similarity = row["Similarity"]
            if similarity >= self.threshold:
                matches.append(
                    ValueMatch(
                        source_attribute,
                        target_attribute,
                        source_value,
                        target_value,
                        similarity,
                    )
                )

        matches = self._sort_matches(matches)

        return self._fill_missing_matches(
            source_values, matches, source_attribute, target_attribute
        )


def remove_non_string_values(values):
    new_values = [x for x in values if not (isinstance(x, (int, float)) or pd.isna(x))]
    return new_values
