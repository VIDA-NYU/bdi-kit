import difflib
from typing import List

from bdikit.value_matching.base import BaseTopkValueMatcher, ValueMatch


class DiffLibMatcher(BaseTopkValueMatcher):
    """
    Value matching algorithm based on the difflib library.
    """

    def __init__(self, threshold: float = 0):
        """
        Initialize the DiffLibMatcher.

        :param threshold: The threshold for similarity matching.
        """
        self.threshold = threshold

    def match(
        self,
        source_values: List[str],
        target_values: List[str],
        top_k: int,
    ) -> List[ValueMatch]:
        """
        Match source values to target values using difflib.

        :param source_values: The source values to match.
        :param target_values: The target values to match against.
        :return: A list of ValueMatch objects representing the matches.
        """
        matches = []
        for source in source_values:
            source_top_k = []
            best_matches = difflib.get_close_matches(
                source.lower(),
                [val.lower() for val in target_values],
                n=top_k,
                cutoff=self.threshold,
            )

            if best_matches:
                for target in best_matches:
                    similarity = difflib.SequenceMatcher(None, source, target).ratio()
                    source_top_k.append(ValueMatch(source, target, similarity))

                def get_similarity(value_match: ValueMatch) -> float:
                    return value_match.similarity

                source_top_k.sort(key=get_similarity, reverse=True)
            matches.extend(source_top_k)

        return matches
