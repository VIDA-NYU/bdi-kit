from typing import List, NamedTuple, TypedDict, Set


class ValueMatch(NamedTuple):
    """
    Represents a match between a source value and a target value with a
    similarity score.
    """

    source_value: str
    target_value: str
    similarity: float


class ValueMatchingResult(TypedDict):
    """
    Represents the result of a value matching operation.
    """

    source: str
    target: str
    matches: List[ValueMatch]
    coverage: float
    unique_values: Set[str]
    unmatch_values: Set[str]


class BaseValueMatcher:
    """
    Base class for value matching algorithms, i.e., algorithms that match
    values from a source domain to values from a target domain.
    """

    def match_values(
        self, source_values: List[str], target_values: List[str]
    ) -> List[ValueMatch]:
        raise NotImplementedError("Subclasses must implement this method")


class BaseTopkValueMatcher(BaseValueMatcher):
    def rank_value_matches(
        self, source_values: List[str], target_values: List[str], top_k: int
    ) -> List[ValueMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def match_values(
        self, source_values: List[str], target_values: List[str]
    ) -> List[ValueMatch]:
        matches = self.rank_value_matches(source_values, target_values, 1)

        return matches
