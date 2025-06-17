import numpy as np
from collections import defaultdict
from typing import List, NamedTuple, Dict, Any


class ValueMatch(NamedTuple):
    """
    Represents a match between a source value and a target value with a
    similarity score given a source and target attributes.
    """

    source_attribute: str
    target_attribute: str
    source_value: Any
    target_value: Any
    similarity: float


class BaseValueMatcher:
    """
    Base class for value matching algorithms, i.e., algorithms that match
    values from a source domain to values from a target domain.
    """

    def match_values(
        self,
        source_values: List[Any],
        target_values: List[Any],
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self,
        source_values: List[Any],
        matches: List[ValueMatch],
        source_attribute: str,
        target_attribute: str,
        default_unmatched: Any = np.nan,
    ) -> List[ValueMatch]:
        source_values_set = set(source_values)

        for match in matches:
            if match.source_value in source_values_set:
                source_values_set.remove(match.source_value)

        # Fill missing matches with the default unmatched value
        for source_value in source_values_set:
            matches.append(
                ValueMatch(
                    source_attribute,
                    target_attribute,
                    source_value,
                    default_unmatched,
                    default_unmatched,
                )
            )

        return matches

    def _sort_matches(self, matches: List[ValueMatch]) -> List[ValueMatch]:
        # Group matches by source_value
        grouped_matches = defaultdict(list)
        for match in matches:
            grouped_matches[match.source_value].append(match)

        # Sort each group by similarity
        ordered_groups = [
            sorted(group, key=lambda x: x.similarity, reverse=True)
            for group in grouped_matches.values()
        ]

        # Sort the groups by maximum similarity
        ordered_groups = sorted(
            ordered_groups, key=lambda x: x[0].similarity, reverse=True
        )

        # Flatten the sorted groups into a single list
        sorted_matches = [item for group in ordered_groups for item in group]

        return sorted_matches

    @staticmethod
    def sort_multiple_matches(matches: List[ValueMatch]) -> List[ValueMatch]:
        # Group by source_attribute and target_attribute
        grouped = {}
        for match in matches:
            key = (match.source_attribute, match.target_attribute)
            grouped.setdefault(key, []).append(match)

        # Create a list of groups with their maximum similarity
        groups_with_similarity = [
            (group, group[0].similarity) for group in grouped.values()
        ]

        # Sort the groups by maximum similarity
        ordered_groups = sorted(
            groups_with_similarity, key=lambda x: x[1], reverse=True
        )

        # Flatten the sorted groups into a single list
        sorted_matches = [item for group, _ in ordered_groups for item in group]

        return sorted_matches


class BaseTopkValueMatcher(BaseValueMatcher):
    def rank_value_matches(
        self,
        source_values: List[Any],
        target_values: List[Any],
        top_k: int,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:
        raise NotImplementedError("Subclasses must implement this method")

    def match_values(
        self,
        source_values: List[Any],
        target_values: List[Any],
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ValueMatch]:
        matches = self.rank_value_matches(
            source_values, target_values, 1, source_context, target_context
        )

        return matches
