import pandas as pd
from typing import Dict, Optional
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
from bdikit.models.contrastive_learning.cl_api import (
    DEFAULT_CL_MODEL,
)
from bdikit.mapping_algorithms.column_mapping.topk_matchers import (
    TopkColumnMatcher,
    CLTopkColumnMatcher,
)


class BaseSchemaMatcher:
    def map(self, dataset: pd.DataFrame, global_table: pd.DataFrame) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement this method")

    def _fill_missing_matches(
        self, dataset: pd.DataFrame, matches: Dict[str, str]
    ) -> Dict[str, str]:
        for column in dataset.columns:
            if column not in matches:
                matches[column] = ""
        return matches


class ValentineSchemaMatcher(BaseSchemaMatcher):
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


class SimFloodSchemaMatcher(ValentineSchemaMatcher):
    def __init__(self):
        super().__init__(SimilarityFlooding())


class ComaSchemaMatcher(ValentineSchemaMatcher):
    def __init__(self):
        super().__init__(Coma())


class CupidSchemaMatcher(ValentineSchemaMatcher):
    def __init__(self):
        super().__init__(Cupid())


class DistributionBasedSchemaMatcher(ValentineSchemaMatcher):
    def __init__(self):
        super().__init__(DistributionBased())


class JaccardSchemaMatcher(ValentineSchemaMatcher):
    def __init__(self):
        super().__init__(JaccardDistanceMatcher())


class GPTSchemaMatcher(BaseSchemaMatcher):
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

    def get_column_type(
        self, context: str, labels: str, m: int = 10, model: str = "gpt-4-turbo-preview"
    ):
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


class ContrastiveLearningSchemaMatcher(BaseSchemaMatcher):
    def __init__(self, model_name: str = DEFAULT_CL_MODEL):
        self.topk_matcher = CLTopkColumnMatcher(model_name=model_name)

    def map(self, dataset: pd.DataFrame, global_table: pd.DataFrame):
        topk_matches = self.topk_matcher.get_recommendations(
            dataset, global_table, top_k=1
        )
        matches = {}
        for column, top_k_match in zip(dataset.columns, topk_matches):
            candidate = top_k_match["top_k_columns"][0][0]
            if candidate in global_table.columns:
                matches[column] = candidate
        return self._fill_missing_matches(dataset, matches)


class TwoPhaseSchemaMatcher(BaseSchemaMatcher):

    def __init__(
        self,
        top_k: int = 20,
        top_k_matcher: Optional[TopkColumnMatcher] = None,
        schema_matcher: BaseSchemaMatcher = SimFloodSchemaMatcher(),
    ):
        if top_k_matcher is None:
            self.api = CLTopkColumnMatcher(DEFAULT_CL_MODEL)
        elif isinstance(top_k_matcher, TopkColumnMatcher):
            self.api = top_k_matcher
        else:
            raise ValueError(
                f"Invalid top_k_matcher type: {type(top_k_matcher)}. "
                "Must be a subclass of {TopkColumnMatcher.__name__}"
            )

        self.schema_matcher = schema_matcher
        self.top_k = top_k

    def map(
        self,
        dataset: pd.DataFrame,
        global_table: pd.DataFrame,
    ):
        topk_column_matches = self.api.get_recommendations(
            dataset, global_table, self.top_k
        )

        matches = {}
        for column, scope in zip(dataset.columns, topk_column_matches):
            candidates = [
                cand[0]
                for cand in scope["top_k_columns"]
                if cand[0] in global_table.columns
            ]
            reduced_dataset = dataset[[column]]
            reduced_global_table = global_table[candidates]
            partial_matches = self.schema_matcher.map(
                reduced_dataset, reduced_global_table
            )

            if column in partial_matches:
                matches[column] = partial_matches[column]

        return self._fill_missing_matches(dataset, matches)
