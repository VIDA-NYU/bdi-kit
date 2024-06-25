import unittest
import pandas as pd
from bdikit.mapping_algorithms.value_mapping.algorithms import (
    TFIDFAlgorithm,
    EditAlgorithm,
)


class ValueMatchingAlgorithmsTest(unittest.TestCase):

    def test_tfidf_value_matching(self):
        # given
        current_values = ["Red Apple", "Banana", "Oorange", "Strawberry"]
        target_values = ["apple", "banana", "orange", "kiwi"]

        tfidf_matcher = TFIDFAlgorithm()

        # when
        matches = tfidf_matcher.match(current_values, target_values)

        # then
        self.assertEqual(len(matches), 3)

        mapped_matches = {match[0]: (match[1], match[2]) for match in matches}
        self.assertNotIn("Strawberry", mapped_matches)
        self.assertEqual(mapped_matches["Red Apple"][0], "apple"),
        self.assertEqual(mapped_matches["Banana"][0], "banana"),
        self.assertEqual(mapped_matches["Oorange"][0], "orange")

        scores = [match[2] for match in matches]
        self.assertTrue(all(score > 0.8 for score in scores))

    def test_edit_distance_value_matching(self):
        # given
        current_values = ["Red Apple", "Banana", "Oorange", "Strawberry"]
        target_values = ["apple", "bananana", "orange", "kiwi"]

        edit_distance_matcher = EditAlgorithm()

        # when
        matches = edit_distance_matcher.match(
            current_values, target_values, threshold=0.5
        )

        # then
        self.assertEqual(len(matches), 3)

        mapped_matches = {match[0]: (match[1], match[2]) for match in matches}
        self.assertNotIn("Strawberry", mapped_matches)
        self.assertEqual(mapped_matches["Red Apple"][0], "apple"),
        self.assertEqual(mapped_matches["Banana"][0], "bananana"),
        self.assertEqual(mapped_matches["Oorange"][0], "orange")

        scores = [match[2] for match in matches]
        self.assertTrue(all(score > 0.5 for score in scores))
