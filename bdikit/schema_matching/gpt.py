import pandas as pd
from openai import OpenAI
from bdikit.schema_matching.base import BaseOne2oneSchemaMatcher


class GPT(BaseOne2oneSchemaMatcher):
    def __init__(self):
        self.client = OpenAI()

    def get_one2one_match(self, source: pd.DataFrame, target: pd.DataFrame):
        target_columns = target.columns
        labels = ", ".join(target_columns)
        candidate_columns = source.columns
        mappings = {}
        for column in candidate_columns:
            col = source[column]
            values = col.drop_duplicates().dropna()
            if len(values) > 15:
                rows = values.sample(15).tolist()
            else:
                rows = values.tolist()
            serialized_input = f"{column}: {', '.join([str(row) for row in rows])}"
            context = serialized_input.lower()
            column_types = self.get_column_type(context, labels)
            for column_type in column_types:
                if column_type in target_columns:
                    mappings[column] = column_type
                    break
        return self._fill_missing_matches(source, mappings)

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
