import pandas as pd
import warnings
import json_repair
from litellm import completion
from typing import List, Dict
from bdikit.utils import get_additional_context
from bdikit.schema_matching.base import BaseTopkSchemaMatcher, ColumnMatch


class LLM(BaseTopkSchemaMatcher):
    """A schema matcher that uses LLM to match columns based on their similarity."""

    def __init__(self, model_name="openai/gpt-4o-mini", **model_kwargs):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.llm_attempts = 5

    def _sample_values(self, column, max_samples=5):
        values = column.drop_duplicates().dropna()
        if len(values) > max_samples:
            return values.sample(max_samples).tolist()
        else:
            return values.tolist()

    def rank_schema_matches(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        top_k: int,
        source_context: Dict[str, str] = None,
        target_context: Dict[str, str] = None,
    ) -> List[ColumnMatch]:
        matches = []

        additional_source_cxt = get_additional_context(source_context, "source")
        additional_target_cxt = get_additional_context(target_context, "target")
        additional_context = additional_source_cxt + additional_target_cxt

        target_cols = [
            "Column: "
            + target_col
            + ", Sample values: ["
            + ",".join(self._sample_values(target[target_col]))
            + "]"
            for target_col in target.columns
        ]

        for source_col in source.columns:
            cand = (
                "Column: "
                + source_col
                + ", Sample values: ["
                + ",".join(self._sample_values(source[source_col]))
                + "]"
            )

            targets = "\n".join(target_cols)

            attempts = 0
            while True:
                if attempts >= self.llm_attempts:
                    warnings.warn(
                        f"Failed to parse response after {self.llm_attempts} attempts. Skipping.",
                        UserWarning,
                    )
                    refined_match = []
                    break

                refined_match = self._get_matches(cand, targets, additional_context)
                refined_match = self._parse_matches(refined_match)
                attempts += 1

                if refined_match is not None:
                    break

            sorted_matches = sorted(refined_match, key=lambda x: x[1], reverse=True)
            for target_col, score in sorted_matches[:top_k]:
                matches.append(ColumnMatch(source_col, target_col, score))

        matches = self._sort_ranked_matches(matches)

        return self._fill_missing_matches(source, matches)

    def _get_prompt(self, cand, targets, additional_context):
        prompt = f"""Given a candidate column and a list of target columns, judge the similarity between the candidate and each target column. Return a JSON array of objects, each with 'column' (the target column name) and 'score' (a float between 0 and 1, two decimals, 1 is most similar). {additional_context}
                    Do NOT provide any other output text or explanation. Only provide the JSON array.
                    Example:
                    Candidate Column: Column: EmployeeID, Sample values: [100, 101, 102]
                    Target Schemas:
                    Column: WorkerID, Sample values: [100, 101, 102]
                    Column: EmpCode, Sample values: [001, 002, 003]
                    Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']
                    Response: [
                        {{"column": "WorkerID", "score": 0.95}},
                        {{"column": "EmpCode", "score": 0.30}},
                        {{"column": "StaffName", "score": 0.05}}
                    ]

                    Candidate Column: {cand}

                    Target Schemas:
                    {targets}

                    Response: """

        return prompt

    def _get_matches(self, cand, targets, additional_context):
        prompt = self._get_prompt(cand, targets, additional_context)
        messages = [
            {
                "role": "system",
                "content": "You are an AI trained to perform schema matching by providing column similarity scores.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = completion(
            model=self.model_name,
            messages=messages,
            **self.model_kwargs,
        )
        matches = response.choices[0].message.content

        return matches

    def _parse_matches(self, refined_match):
        try:
            matches_json = json_repair.loads(refined_match)
            matched_columns = []
            for entry in matches_json:
                schema_name = entry.get("column")
                score = float(entry.get("score", 0))
                matched_columns.append((schema_name, score))
            return matched_columns
        except Exception as e:
            warnings.warn(
                f"Error parsing JSON response: {e}\nRaw response: {refined_match}",
                UserWarning,
            )
            return None
