from openai import OpenAI
import tiktoken
import json
import pandas as pd
from tqdm import tqdm

class CTA:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        
    def num_tokens_from_string(self, string, encoding_name="gpt-4-turbo-preview"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    # Reply None if none of the classes are applicable.
    def get_column_type(self, context, labels, m, model="gpt-4-turbo-preview"):
        col_type = self.client.chat.completions.create(model=model,
        messages=[
                {
                    "role": "system", 
                    "content": "You are an assistant for column matching."},
                {
                    "role": "user", 
                    "content": """ Please select the top """ + str(m) +  """ class from """ + labels + """ which best describes the context. The context is defined by the column name followed by its respective values. Please respond only with the name of the classes separated by semicolon.
                    \n CONTEXT: """ + context +  """ \n RESPONSE: \n"""},
            ],
        temperature=0.3)
        col_type_content = col_type.choices[0].message.content
        return col_type_content
    
    
def run_cta():
    api_key = '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'
    annotator = CTA(api_key)

    k = 10
    starmie = False
    m = 5

    file_name = f"top_{k}_starmie_results.json" if starmie else f"top_{k}_results.json"

    # with open("./results-cl/" + file_name, "r") as file:
    with open(file_name, "r") as file:
        json_data = json.load(file)

    total_matches = 0
    total_possible = 0
    table_path = 'data/tables'
    results = []

    for result in tqdm(json_data):
        table = pd.read_csv(f'{table_path}/{result["Table name"]}.csv')
        candidate_column = result["Candidate column"]
        top_k_columns = result["Top k columns"]
        labels = ', '.join(top_k_columns)
        col = table[candidate_column]
        values = col.drop_duplicates().dropna()
        if len(values) > 15:
            rows = values.sample(15).tolist()
        else:
            rows = values.tolist()
        serialized_input = f"{candidate_column}: {', '.join([str(row) for row in rows])}"
        context = serialized_input.lower()
        col_type = annotator.get_column_type(context, labels, m)
        match = {
            "Table name": result["Table name"],
            "Candidate column": candidate_column,
            "Generated GDC variable name": col_type,
            "Ground truth GDC variable column": result["Ground truth column"]
        }
        results.append(match)
        
        total_possible += 1
        if result["Ground truth column"] in col_type:
            total_matches += 1
        
        # print(f"{candidate_column}: {col_type}")
        
    print("Precision: ", total_matches / total_possible)

    with open(f"cta_{m}_{file_name}", "w") as file:
        json.dump(results, file, indent=4)
    print(f"CTA results saved to cta_{m}_{file_name}")


if __name__ == "__main__":
    run_cta()