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


class BaseColumnMappingAlgorithm:
    def __init__(self, dataset, global_table):
        self._dataset = dataset
        self._global_table = global_table

    def map(self) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement this method")


class ValentineColumnMappingAlgorithm(BaseColumnMappingAlgorithm):
    def __init__(self, dataset, global_table, matcher: BaseMatcher):
        super().__init__(dataset, global_table)
        self.matcher = matcher

    def map(self) -> Dict[str, str]:
        matches: MatcherResults = valentine_match(
            self._dataset, self._global_table, self.matcher
        )
        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate
        return mappings


class SimFloodAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table, SimilarityFlooding())


class ComaAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table, Coma())


class CupidAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table, Cupid())


class DistributionBasedAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table, DistributionBased())


class JaccardDistanceAlgorithm(ValentineColumnMappingAlgorithm):
    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table, JaccardDistanceMatcher())


class GPTAlgorithm(BaseColumnMappingAlgorithm):
    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
        self.client = OpenAI()

    def map(self):
        global_columns = self._global_table.columns
        labels = ", ".join(global_columns)
        candidate_columns = self._dataset.columns
        mappings = {}
        for column in candidate_columns:
            col = self._dataset[column]
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
        return mappings

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
