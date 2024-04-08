import os
import sys
import jellyfish

import ast
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from table_matching.value_matcher import EmbeddingMatcher


class GDCScoringInterface:
    @staticmethod
    def compute_col_values_score(values, choices):
        """
        This computes the score between the input values and the candidate gdc values.

        :param values: list, the input enum values from actual data column
        :param choices: list, the enum values from GDC schema

        :return: score: float, a score between 0 and 1
        """
        pass

    @staticmethod
    def compute_col_name_score(col_name, candidate_col_name):
        """
        This computes the score between the input column name and the candidate gdc column name.

        :param col_name: str, the input column name
        :param candidate_col_name: str, the candidate gdc column name

        :return: score: float, a score between 0 and 1
        """
        pass


# this is a sample implementation of the GDCScoringInterface using Jaro similarity
class JaroScore(GDCScoringInterface):
    scorer_name = "jaro"  # must have a scorer_name attribute, and should be unique!

    @staticmethod
    def compute_col_values_score(values, choices):
        """
        Compute the Jaro similarity score between the input values and the choices.
        For each value, it will find the maximum Jaro similarity score with the choices
        and then return the average score.

        :param values: list, the input values
        :param choices: list, the choices to compare with from GDC enums
        :return: score: float, the average Jaro similarity score
        """
        score = 0
        for value in values:
            score += max(
                [jellyfish.jaro_similarity(value, choice) for choice in choices]
            )
        return score / len(values)

    @staticmethod
    def compute_col_name_score(col_name, candidate_col_name):
        return jellyfish.jaro_similarity(col_name, candidate_col_name)

    
class EmbeddingScore(GDCScoringInterface):
    scorer_name = "embedding"
    value_matcher = EmbeddingMatcher()
        
    
    @staticmethod
    def compute_col_values_score(values, choices):
        score = 0
        for value in values:
            EmbeddingScore.value_matcher.model.match([value], choices, top_n=1)
            score += EmbeddingScore.value_matcher.model.get_matches().iloc[0]["Similarity"]
        return score / len(values)
    
    @staticmethod
    def compute_col_name_score(col_name, candidate_col_name):
        EmbeddingScore.value_matcher.model.match([col_name], [candidate_col_name], top_n=1)
        return EmbeddingScore.value_matcher.model.get_matches().iloc[0]["Similarity"]


class GPTHelper:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def ask_by_prompt(self, prompt, model="gpt-4-turbo-preview"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an assistant for column type annotation."},
                {
                    "role": "user", 
                    "content": prompt},
            ],
            temperature=0.3)
        response_message = completion.choices[0].message.content
        return response_message
    
    def ask_cta(self, labels, context):
        prompt = f""" Please select the class from {labels} which best describes the context. The context is defined by the column name followed by its respective values. Please respond only with the name of the class. Reply None if none of the classes are applicable.
                    \n CONTEXT: """ + context +  """ \n RESPONSE: \n"""
        
        return self.ask_by_prompt(prompt)
    
