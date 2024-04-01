import ast
from openai import OpenAI
from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, Embeddings
from flair.embeddings import TransformerWordEmbeddings


class BaseMatcher():

    def __init__(self, *args):
        pass
    
    def match(self, current_values, target_values, threshold=0.8):
        self.model.match(current_values, target_values)
        match_results = self.model.get_matches()
        match_results.sort_values(by='Similarity', ascending=False, inplace=True)

        matches = []
        #used_values = set()
        for _, row in match_results.iterrows():
            current_value = row['From']
            target_value = row['To']
            similarity = row['Similarity']
            if similarity >= threshold:
                matches.append((current_value, target_value, similarity))

        return matches


class TFIDFMatcher(BaseMatcher):

    def __init__(self):
        method = TFIDF(min_similarity=0)
        self.model = PolyFuzz(method)
    
    def match(self, current_values, target_values, threshold=0.8):
        matches = super().match(current_values, target_values, threshold)

        return matches


class EditMatcher(BaseMatcher):

    def __init__(self):
        method = EditDistance(n_jobs=-1)
        self.model = PolyFuzz(method)
    
    def match(self, current_values, target_values, threshold=0.8):
        matches = super().match(current_values, target_values, threshold)

        return matches


class EmbeddingMatcher(BaseMatcher):

    def __init__(self, model_path='bert-base-multilingual-cased'):
        embeddings = TransformerWordEmbeddings(model_path)
        method = Embeddings(embeddings, min_similarity=0, model_id='embedding_model')
        self.model = PolyFuzz(method)
    
    def match(self, current_values, target_values, threshold=0.8):
        matches = super().match(current_values, target_values, threshold)

        return matches


class LLMMatcher(BaseMatcher):

    def __init__(self):
        self.client = OpenAI()
    
    def match(self, current_values, target_values, threshold=0.8):
        completion = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an intelligent system designed for mapping values from a source list to a target list."\
            "These values belong to the medical domain, and the target list contains values in the Genomics Data Commons (GDC) format."},
            {"role": "user", "content": f"The source list is: {current_values}"\
            "The target list is: {target_values}"\
            "Find the pairs and return a list of Python tuples with a similarity value, between 0 and 1, with 1 indicating the highest similarity"\
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python list. For example [(source value, target value, 0.8)]"}
        ]
        )
        response_message = completion.choices[0].message.content
        matches = []
        try:
            matches = ast.literal_eval(response_message)
        except:
            print(f'Errors parsing response: {response_message}')

        return matches
    


if __name__ == '__main__':
    from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
    to_list = ["apple", "apples", "mouse"]
    matcher = TFIDFMatcher()
    matches = matcher.match(from_list, to_list)
    print(matches)
