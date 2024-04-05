import tiktoken
from openai import OpenAI
import json
import pandas as pd
import os
from tqdm import tqdm

class CTA:
    def __init__(self, api_key, input_dir, output_dir, labels):
        self.api_key = api_key
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.client = OpenAI(api_key=self.api_key)
        self.labels = labels
        os.makedirs(self.output_dir, exist_ok=True)
        
    def num_tokens_from_string(self, string, encoding_name="gpt-4-turbo-preview"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_column_type(self, context, model):
        # TODO
        # ask GPT what is the type if it's not matched to any of the classes in GDC
        # RAG from GDC descriptions to find one from None
        col_type = self.client.chat.completions.create(model=model,
        messages=[
                {
                    "role": "system", 
                    "content": "You are an assistant for column type annotation."},
                {
                    "role": "user", 
                    "content": """ Please select the class from """ + self.labels + """ which best describes the context. The context is defined by the column name followed by its respective values. Please respond only with the name of the class. Reply None if none of the classes are applicable.
                    \n CONTEXT: """ + context +  """ \n RESPONSE: \n"""},
            ],
        temperature=0.3)
        col_type_content = col_type.choices[0].message.content
        return col_type_content

    def annotate(self, model="gpt-4-turbo-preview"):
        results = {}
        for filename in tqdm(os.listdir(self.input_dir), desc="Annotating columns"):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.input_dir, filename)
                df = pd.read_csv(file_path)
                
                for col_index, col_name in enumerate(df.columns):
                    # Select 5 random rows from the column
                    values = df[col_name].drop_duplicates().dropna()
                    if len(values) > 15:
                        rows = values.sample(15).tolist()
                    else:
                        rows = values.tolist()
                    serialized_input = f"{col_name}: {', '.join([str(row) for row in rows])}"

                    # preprocess
                    context = serialized_input.lower().replace("-", "_")
                    # Column type annotation
                    # num_tokens = self.num_tokens_from_string(context)
                    # if num_tokens > 128000:
                    #     context = context[:128000]
                    col_type = self.get_column_type(context, model)
                    
                    table_name = filename.split("_")[0]
                    if table_name not in results:
                        results[table_name] = []
                    results[table_name].append((col_name, col_type))
                    
        with open(os.path.join(self.output_dir, 'column_types.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        for key, value in results.items():
            df = pd.DataFrame(value, columns=["column_name", "generated_column_type"])
            df.to_csv(os.path.join(self.output_dir, f"{key}_column_types.csv"), index=False)

def main(api_key, input_dir, output_dir):    
    gdc_variable_names = []
    with open('./table-union/cta/gdc_format_variable_names_in_gt.txt', 'r') as f:
        for line in f:
            gdc_variable_names.append(line.strip())
    context = ', '.join(gdc_variable_names)
    annotater = CTA(api_key, input_dir, output_dir, context)
    annotater.annotate()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for the columns in the tables in the input directory.")
    parser.add_argument("--input_dir", type=str, default="./data/extracted-tables", help="The directory containing the tables.")
    parser.add_argument("--output_dir", type=str, default="./table-union/cta", help="The directory to store the embeddings.")
    args = parser.parse_args()
    api_key = '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' #TODO: Replace with your own OpenAI API key
    
    main(api_key, args.input_dir, args.output_dir)
    
    # # read Dou_column_types.csv and Dou.csv
    # dou_col_types = pd.read_csv('./table-union/cta/Dou_column_types.csv')
    # dou = pd.read_csv('./table-union/cta/Dou.csv')
    # # merge on dou_col_types['column_name'] == dou['original_paper_variable_names'], null as None
    # dou = pd.merge(dou, dou_col_types, how='inner', left_on='original_paper_variable_names', right_on='column_name')
    # dou.to_csv('./table-union/cta/dou_inner.csv', index=False)