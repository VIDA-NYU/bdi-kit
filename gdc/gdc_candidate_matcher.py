import logging

import jellyfish
import pandas as pd

from gdc_api import GDCSchema
from gdc_scoring_interface import JaroScore

logger = logging.getLogger(__name__)


class GDCCandidateMatcher:
    """
    GDCCandidateMatcher class is used to match the input values
    and column names with the GDC schema.

    Example:
    ```
    matcher = GDCCandidateMatcher(subschemas=['demographic', 'diagnosis', 'sample'])
    enums = matcher.extract_enums()
    df_enums = matcher.extract_df_enums(df)
    embedded_scores = matcher.compute_embedded_col_values_name_score(df_enums)
    matches = matcher.parse_df(df)
    ```

    :param subschemas: list, the list of subschemas to match with
    :param scorers: list, the list of scorers to compute the similarity score
        scorer object must follows the GDCScoringInterface
    """

    def __init__(self, subschemas=None):
        self.schema_api = GDCSchema(subschemas=subschemas)
        self.scorers = set()

    def add_scorer(self, scorer):
        self.scorers.add(scorer)

    def extract_enums(self):
        enums = {}
        for subschema, values in self.schema_api.get_schema().items():
            for col_name, properties in values["properties"].items():
                if "enum" in properties:
                    name = f"{subschema}::{col_name}"
                    if name not in enums:
                        enums[name] = []
                    enums[name].extend([value.lower() for value in properties["enum"]])
        for col_name, values in enums.items():
            enums[col_name] = list(set(values))

        return enums

    @staticmethod
    def extract_df_enums(df):
        df_enums = df.apply(pd.unique).to_dict()

        for key, values in df_enums.items():
            df_enums[key] = list(set([str(value).lower() for value in values]))
            if "nan" in df_enums[key]:
                df_enums[key].remove("nan")
        return df_enums

    def compute_embedded_col_values_name_score(self, df_enums):
        """
        Compute Jaro Score (see: compute_jaro_score) and column name similarity score.
        Output is sorted by Jaro Score and then by column name similarity score.

        :param df_enums: dict, the input column names and their unique values (should be lower cased)
        :param candidate_enums: dict, the candidate column names and their unique values (should be lower cased)
        :return: embedded_scores: dict, the embedded scores of column names and their candidate column names
        """
        candidate_enums = self.extract_enums()
        embedded_scores = {}
        for col_name, values in df_enums.items():
            scores = {}
            for candidate, choices in candidate_enums.items():
                scores[candidate] = {}
                for scorer in self.scorers:
                    scores[candidate][
                        f"{scorer.scorer_name}-values"
                    ] = scorer.compute_col_values_score(values, choices)
                    scores[candidate][
                        f"{scorer.scorer_name}-name"
                    ] = scorer.compute_col_name_score(
                        col_name, candidate
                    )  # note that candidate is the subschema::col_name

            scores = {
                k: v
                for k, v in sorted(
                    scores.items(),
                    key=lambda item: list(item[1].values()),
                    reverse=True,
                )
            }
            # print(f"{col_name}: {list(scores.keys())[0]}")
            embedded_scores[col_name] = scores
        return embedded_scores

    def parse_df(self, df, number_of_candidates=1, as_mapping=False):
        df_enums = self.extract_df_enums(df)
        enums = self.extract_enums()
        embedded_scores = self.compute_embedded_col_values_name_score(df_enums)
        matches = {}
        if as_mapping:
            gdc_data = {}
        for col_name, scores in embedded_scores.items():
            if as_mapping:
                matches[col_name] = list(scores.keys())[0]
                gdc_data[list(scores.keys())[0]] = enums[list(scores.keys())[0]]
                continue
            matches[col_name] = []
            candidates_count = 0
            for candidate, score in scores.items():
                if candidates_count >= number_of_candidates:
                    break
                self.schema_api.get_properties_by_gdc_candidate(candidate)

                candidate_score = {
                    "candidate": candidate,
                    "type": self.schema_api.get_gdc_col_type(),
                    "values": self.schema_api.get_gdc_col_values(),
                }
                for scorer_name, value in score.items():
                    candidate_score[scorer_name] = value

                matches[col_name].append(candidate_score)

                candidates_count += 1
        if as_mapping:
            return matches, gdc_data
        return matches

