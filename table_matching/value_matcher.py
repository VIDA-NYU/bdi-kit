from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, Embeddings


class BaseMatcher():

    def __init__(self, *args):
        pass
    
    def match(self, current_values, target_values, threshold=0.8):
        self.model.match(current_values, target_values)
        match_results = self.model.get_matches()

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
    

if __name__ == '__main__':
    from_list = ["apple", "apples", "appl", "recal", "house", "similarity"]
    to_list = ["apple", "apples", "mouse"]
    matcher = TFIDFMatcher()
    matches = matcher.match(from_list, to_list)
    print(matches)
