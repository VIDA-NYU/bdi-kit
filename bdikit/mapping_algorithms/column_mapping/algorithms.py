import pandas as pd
from typing import Dict
from valentine import valentine_match
from valentine.algorithms import (
    SimilarityFlooding,
    Coma,
    Cupid,
    DistributionBased,
    JaccardDistanceMatcher,
    BaseMatcher,
)
from valentine.algorithms.matcher_results import MatcherResults
from openai import OpenAI
from bdikit.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import (
    ContrastiveLearningAPI,
)
from bdikit.download import get_cached_model_or_download


class BaseColumnMappingAlgorithm:
    def map(self, dataset: pd.DataFrame, global_table: pd.DataFrame) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self, dataset: pd.DataFrame, matches: Dict[str, str]
    ) -> Dict[str, str]:
        for column in dataset.columns:
            if column not in matches:
                matches[column] = ""
        return matches


class ValentineColumnMappingAlgorithm(BaseColumnMappingAlgorithm):
    def __init__(self, matcher: BaseMatcher):
        self.matcher = matcher

    def map(self, dataset: pd.DataFrame, global_table: pd.DataFrame) -> Dict[str, str]:
        matches: MatcherResults = valentine_match(dataset, global_table, self.matcher)
        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate
        return self._fill_missing_matches(dataset, mappings)


class SimFloodAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self):
        super().__init__(SimilarityFlooding())


class ComaAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self):
        super().__init__(Coma())


class CupidAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self):
        super().__init__(Cupid())


class DistributionBasedAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self):
        super().__init__(DistributionBased())


class JaccardDistanceAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self):
        super().__init__(JaccardDistanceMatcher())


class GPTAlgorithm(BaseColumnMappingAlgorithm):
    def __init__(self):
        self.client = OpenAI()

    def map(self, dataset: pd.DataFrame, global_table: pd.DataFrame):
        global_columns = global_table.columns
        labels = ", ".join(global_columns)
        candidate_columns = dataset.columns
        mappings = {}
        for column in candidate_columns:
            col = dataset[column]
            values = col.drop_duplicates().dropna()
            if len(values) > 15:
                rows = values.sample(15).tolist()
            else:
                rows = values.tolist()
            serialized_input = f"{column}: {', '.join([str(row) for row in rows])}"
            context = serialized_input.lower()
            column_types = self.get_column_type(context, labels)
            for column_type in column_types:
                if column_type in global_columns:
                    mappings[column] = column_type
                    break
        return self._fill_missing_matches(dataset, mappings)

    def get_column_type(self, context, labels, m=10, model="gpt-4-turbo-preview"):
        messages = [
            {"role": "system", "content": "You are an assistant for column matching."},
            {
                "role": "user",
                "content": """ Please select the top """
                + str(m)
                + """ class from """
                + labels
                + """ which best describes the context. The context is defined by the column name followed by its respective values. Please respond only with the name of the classes separated by semicolon.
                    \n CONTEXT: """
                + context
                + """ \n RESPONSE: \n""",
            },
        ]
        col_type = self.client.chat.completions.create(
            model=model, messages=messages, temperature=0.3
        )
        col_type_content = col_type.choices[0].message.content
        return col_type_content.split(";")


class ContrastiveLearningAlgorithm(BaseColumnMappingAlgorithm):
    def __init__(self):
        model_path = get_cached_model_or_download("cl-reducer-v0.1")
        self.api = ContrastiveLearningAPI(model_path=model_path, top_k=20)

    def map(self, dataset: pd.DataFrame, global_table: pd.DataFrame):
        union_scopes, scopes_json = self.api.get_recommendations(dataset)
        matches = {}
        for column, scope in zip(dataset.columns, scopes_json):
            candidate = scope["Top k columns"][0][0]
            if (
                candidate in global_table.columns
            ):  # this check protects against the case where the candidate generated from the model is not in the global table
                matches[column] = candidate
        return self._fill_missing_matches(dataset, matches)


class TwoPhaseMatcherAlgorithm(BaseColumnMappingAlgorithm):
    def __init__(self):
        model_path = get_cached_model_or_download("cl-reducer-v0.1")
        self.api = ContrastiveLearningAPI(model_path=model_path, top_k=20)

    def map(
        self,
        dataset: pd.DataFrame,
        global_table: pd.DataFrame,
        algorithm: BaseColumnMappingAlgorithm = SimFloodAlgorithm(),
    ):
        union_scopes, scopes_json = self.api.get_recommendations(dataset)
        matches = {}
        for column, scope in zip(dataset.columns, scopes_json):
            candidates = [
                cand[0]
                for cand in scope["Top k columns"]
                if cand[0] in global_table.columns
            ]
            reduced_dataset = dataset[[column]]
            reduced_global_table = global_table[candidates]
            partial_matches = algorithm.map(reduced_dataset, reduced_global_table)

            if len(partial_matches.keys()) > 0:
                candidate_col = next(iter(partial_matches))
                target_col = partial_matches[candidate_col]
                matches[candidate_col] = target_col

        return self._fill_missing_matches(dataset, matches)
