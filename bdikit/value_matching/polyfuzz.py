import flair
import torch
from rapidfuzz import fuzz
from polyfuzz import PolyFuzz as PolyFuzzLib
from typing import List, Callable, Tuple
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
        source_values: List[str],
        target_values: List[str],
        top_k: int,
    ) -> List[ValueMatch]:

        if len(target_values) == 0:
            matches = [ValueMatch(source, None, 0.0) for source in source_values]
            return matches

        self.model.method.top_n = top_k
        self.model.match(source_values, target_values)
        match_results = self.model.get_matches()
        match_results.sort_values(by="Similarity", ascending=False, inplace=True)

        matches = []
        for _, row in match_results.iterrows():
            source = row[0]
            top_matches = row[1:]
            indexes = range(0, len(top_matches) - 1, 2)

            for index in indexes:
                target = top_matches[index]
                similarity = top_matches[index + 1]
                if similarity >= self.threshold:
                    matches.append(ValueMatch(source, target, similarity))

        return matches


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
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:

        if len(target_values) == 0:
            matches = [ValueMatch(source, None, 0.0) for source in source_values]
            return matches

        self.model.match(source_values, target_values)
        match_results = self.model.get_matches()
        match_results.sort_values(by="Similarity", ascending=False, inplace=True)

        matches = []
        for _, row in match_results.iterrows():
            source_value = row["From"]
            target_value = row["To"]
            similarity = row["Similarity"]
            if similarity >= self.threshold:
                matches.append(ValueMatch(source_value, target_value, similarity))

        return matches
