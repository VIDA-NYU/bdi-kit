import ast
from openai import OpenAI
from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, Embeddings
from flair.embeddings import TransformerWordEmbeddings
from autofj import AutoFJ
from Levenshtein import ratio
import pandas as pd


class BaseAlgorithm:

    def __init__(self, *args):
        pass

    def match(self, current_values, target_values, threshold=0.8):
        self.model.match(current_values, target_values)
        match_results = self.model.get_matches()
        match_results.sort_values(by="Similarity", ascending=False, inplace=True)
        matches = []

        for _, row in match_results.iterrows():
            current_value = row["From"]
            target_value = row["To"]
            similarity = row["Similarity"]
            if similarity >= threshold:
                matches.append((current_value, target_value, similarity))

        return matches


class TFIDFAlgorithm(BaseAlgorithm):
    def __init__(self):
        method = TFIDF(min_similarity=0)
        self.model = PolyFuzz(method)

    def match(self, current_values, target_values, threshold=0.8):
        matches = super().match(current_values, target_values, threshold)

        return matches


class EditAlgorithm(BaseAlgorithm):
    def __init__(self):
        method = EditDistance(n_jobs=-1)
        self.model = PolyFuzz(method)

    def match(self, current_values, target_values, threshold=0.8):
        matches = super().match(current_values, target_values, threshold)

        return matches


class EmbeddingAlgorithm(BaseAlgorithm):

    def __init__(self, model_path="bert-base-multilingual-cased"):
        embeddings = TransformerWordEmbeddings(model_path)
        method = Embeddings(embeddings, min_similarity=0, model_id="embedding_model")
        self.model = PolyFuzz(method)

    def match(self, current_values, target_values, threshold=0.8):
        matches = super().match(current_values, target_values, threshold)

        return matches


class LLMAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.client = OpenAI()

    def match(self, current_values, target_values, threshold=0.8):
        target_values_set = set(target_values)
        matches = []

        for current_value in current_values:
            completion = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent system that given a term, you have to choose a value from a list that best matches the term. "
                        "These terms belong to the medical domain, and the list contains terms in the Genomics Data Commons (GDC) format.",
                    },
                    {
                        "role": "user",
                        "content": f'For the term: "{current_value}", choose a value from this list {target_values}. '
                        "Return the value from the list with a similarity score, between 0 and 1, with 1 indicating the highest similarity. "
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
                        'Only provide a Python dictionary. For example {"term": "term from the list", "score": 0.8}.',
                    },
                ],
            )

            response_message = completion.choices[0].message.content
            try:
                response_dict = ast.literal_eval(response_message)
                if response_dict["term"] in target_values_set:
                    matches.append(
                        (current_value, response_dict["term"], response_dict["score"])
                    )
            except:
                print(
                    f'Errors parsing response for "{current_value}": {response_message}'
                )

        return matches


class AutoFuzzyJoinAlgorithm(BaseAlgorithm):

    def __init__(self):
        pass

    def match(self, current_values, target_values, threshold=0.8):

        current_values = sorted(list(set(current_values)))
        target_values = sorted(list(set(target_values)))

        df_curr_values = pd.DataFrame(
            {"id": range(1, len(current_values) + 1), "title": current_values}
        )
        df_target_values = pd.DataFrame(
            {"id": range(1, len(target_values) + 1), "title": target_values}
        )

        matches = []
        try:
            autofj = AutoFJ(
                precision_target=threshold,
                join_function_space="autofj_md",
                verbose=True,
            )
            LR_joins = autofj.join(df_curr_values, df_target_values, id_column="id")
            if len(LR_joins) > 0:
                for index, row in LR_joins.iterrows():
                    title_l = row["title_l"]
                    title_r = row["title_r"]
                    similarity = ratio(title_l, title_r)
                    if similarity >= threshold:
                        matches.append((title_l, title_r, similarity))
        except Exception as e:
            return matches
        return matches
