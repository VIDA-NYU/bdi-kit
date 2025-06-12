import os
import re
from openai import OpenAI
from bdikit.schema_matching.base import BaseTopkSchemaMatcher, ColumnMatch


class LLM(BaseTopkSchemaMatcher):
    """A schema matcher that uses LLM to match columns based on their similarity."""

    def __init__(self, llm_model="gpt-4o-mini"):
        self.llm_model = llm_model
        self.client = self._load_client()
        self.llm_attempts = 5

    def _load_client(self):
        if self.llm_model in ["gpt-4-turbo-preview", "gpt-4o-mini"]:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key not found in environment variables.")
            return OpenAI(api_key=api_key)

        else:
            raise ValueError("Invalid model name.")

    def _sample_values(self, column, max_samples=5):
        values = column.drop_duplicates().dropna()
        if len(values) > max_samples:
            return values.sample(max_samples).tolist()
        else:
            return values.tolist()

    def rank_schema_matches(self, source, target, top_k):
        matches = []

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
                    print(
                        f"Failed to parse response after {self.llm_attempts} attempts. Skipping."
                    )
                    refined_match = []
                    break

                refined_match = self._get_matches(cand, targets)
                refined_match = self._parse_matches(refined_match)
                attempts += 1

                if refined_match is not None:
                    break

            sorted_matches = sorted(refined_match, key=lambda x: x[1], reverse=True)
            for target_col, score in sorted_matches[:top_k]:
                matches.append(ColumnMatch(source_col, target_col, score))

        matches = self._sort_ranked_matches(matches)

        return self._fill_missing_matches(source, matches)

    def _get_prompt(self, cand, targets):
        prompt = (
            "From a score of 0.00 to 1.00, please judge the similarity of the candidate column from the candidate table to each target column in the target table. \
All the columns are defined by the column name and a sample of its respective values if available. \
Provide only the name of each target column followed by its similarity score in parentheses, formatted to two decimals, and separated by a semicolon. \
Rank the schema-score pairs by score in descending order. Ensure your response excludes additional information and quotations.\n \
Example:\n \
Candidate Column: \
Column: EmployeeID, Sample values: [100, 101, 102]\n \
Target Schemas: \
Column: WorkerID, Sample values: [100, 101, 102] \
Column: EmpCode, Sample values: [001, 002, 003] \
Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']\n \
Response: WorkerID(0.95); EmpCode(0.30); StaffName(0.05)\n\n \
Candidate Column:"
            + cand
            + "\n\nTarget Schemas:\n"
            + targets
            + "\n\nResponse: "
        )
        return prompt

    def _get_matches(self, cand, targets):
        prompt = self._get_prompt(cand, targets)
        if self.llm_model in [
            "gpt-4-turbo-preview",
            "gpt-4o-mini",
            "llama3.3-70b",
            "meta-llama/Llama-3.3-70B-Instruct",
        ]:
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

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
            )
            matches = response.choices[0].message.content

        else:
            raise ValueError("Invalid model name.")

        return matches

    def _parse_matches(self, refined_match):
        matched_columns = []
        entries = refined_match.split("; ")

        for entry in entries:
            try:
                schema_part, score_part = entry.rsplit("(", 1)
            except ValueError:
                print(f"Error parsing entry: {entry}")
                return None

            try:
                score = float(score_part[:-1])
            except ValueError:
                # Remove all trailing ')'
                score_part = score_part[:-1].rstrip(")")
                try:
                    score = float(score_part)
                except ValueError:
                    cleaned_part = re.sub(
                        r"[^\d\.-]", "", score_part
                    )  # Remove everything except digits, dot, and minus
                    match = re.match(r"^-?\d+\.\d{2}$", cleaned_part)
                    if match:
                        score = float(match.group())
                    else:
                        print("The string does not contain a valid two decimal float.")
                        return None

            schema_name = schema_part.strip()
            matched_columns.append((schema_name, score))

        return matched_columns
