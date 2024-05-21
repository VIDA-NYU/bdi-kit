from valentine import valentine_match
from valentine.algorithms import SimilarityFlooding,Coma,Cupid,DistributionBased,JaccardDistanceMatcher
from openai import OpenAI


class BaseColumnMappingAlgorithm():
    
    def __init__(self, dataset, global_table):
        self._dataset = dataset
        self._global_table = global_table

    def map(self):
        raise NotImplementedError("Subclasses must implement this method")


class SimFlood(BaseColumnMappingAlgorithm):

    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
    
    def map(self):
        matcher = SimilarityFlooding()
        matches = valentine_match(self._dataset, self._global_table, matcher)

        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate        
        return mappings
    
class ComaAlgorithm(BaseColumnMappingAlgorithm):

    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
    
    def map(self):
        matcher = Coma()
        matches = valentine_match(self._dataset, self._global_table, matcher)

        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate        
        return mappings
    
class CupidAlgorithm(BaseColumnMappingAlgorithm):
    
        def __init__(self, dataset, global_table):
            super().__init__(dataset, global_table)
        
        def map(self):
            matcher = Cupid()
            matches = valentine_match(self._dataset, self._global_table, matcher)
    
            mappings = {}
            for match in matches.one_to_one():
                dataset_candidate = match[0][1]
                global_table_candidate = match[1][1]
                mappings[dataset_candidate] = global_table_candidate        
            return mappings
        
class DistributionBasedAlgorithm(BaseColumnMappingAlgorithm):
        
            def __init__(self, dataset, global_table):
                super().__init__(dataset, global_table)
            
            def map(self):
                matcher = DistributionBased()
                matches = valentine_match(self._dataset, self._global_table, matcher)
        
                mappings = {}
                for match in matches.one_to_one():
                    dataset_candidate = match[0][1]
                    global_table_candidate = match[1][1]
                    mappings[dataset_candidate] = global_table_candidate        
                return mappings
            
class JaccardDistanceAlgorithm(BaseColumnMappingAlgorithm):
                
                def __init__(self, dataset, global_table):
                    super().__init__(dataset, global_table)
                
                def map(self):
                    matcher = JaccardDistanceMatcher()
                    matches = valentine_match(self._dataset, self._global_table, matcher)
            
                    mappings = {}
                    for match in matches.one_to_one():
                        dataset_candidate = match[0][1]
                        global_table_candidate = match[1][1]
                        mappings[dataset_candidate] = global_table_candidate        
                    return mappings
            

class GPTAlgorithm(BaseColumnMappingAlgorithm):
    
    def __init__(self, dataset, global_table, api_key="sk-proj-YOUR_API_KEY"):
        super().__init__(dataset, global_table)
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
    
    def map(self):
        global_columns = self._global_table.columns
        labels = ', '.join(global_columns)
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
                    print(f"Column: {column} is of type: {column_type}")
                    break
        return mappings



    def get_column_type(self, context, labels, m=10, model="gpt-4-turbo-preview"):
        messages=[
                {
                    "role": "system", 
                    "content": "You are an assistant for column matching."},
                {
                    "role": "user", 
                    "content": """ Please select the top """ + str(m) +  """ class from """ + labels + """ which best describes the context. The context is defined by the column name followed by its respective values. Please respond only with the name of the classes separated by semicolon.
                    \n CONTEXT: """ + context +  """ \n RESPONSE: \n"""},
            ]
        col_type = self.client.chat.completions.create(model=model,
        messages=messages,
        temperature=0.3)
        col_type_content = col_type.choices[0].message.content
        return col_type_content.split(";")