import os
import pandas as pd
from openai import OpenAI

class EmbeddingsGenerator:
    def __init__(self, api_key, input_dir, output_dir, sample, model="text-embedding-3-small"):
        self.api_key = api_key
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.sample = sample
        os.makedirs(self.output_dir, exist_ok=True)

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding


    def generate_embeddings(self, engine="text-similarity-babbage-001"):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.input_dir, filename)
                df = pd.read_csv(file_path)
                
                for col_index, col_name in enumerate(df.columns):
                    if self.sample:
                        # Select 5 random rows from the column
                        if len(df[col_name].dropna()) < 5:
                            print(f"Not enough data in {col_name} to sample.")
                            continue
                        
                        random_rows = df[col_name].dropna().sample(5).tolist()
                        serialized_input = f"{col_name}: {', '.join([str(row) for row in random_rows])}"
                    else:
                        serialized_input = col_name
                    # preprocess
                    serialized_input = serialized_input.lower().replace(" ", "_").replace("-", "_")
                    # Generate the embedding for the serialized input
                    embedding = self.get_embedding(serialized_input)
                    self._store_embeddings(filename, col_index, embedding)
                    
                    print(f"Generated embeddings for {filename}, column {col_name}.")

    def _store_embeddings(self, filename, col_index, embeddings):
        base_filename = os.path.splitext(filename)[0]
        embeddings_file_path = os.path.join(self.output_dir, f"{base_filename}_{col_index}.pkl")
        
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.to_pickle(embeddings_file_path)

def main(api_key, input_dir, output_dir, sample):    
    embedder = EmbeddingsGenerator(api_key, input_dir, output_dir, sample)
    embedder.generate_embeddings()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for the columns in the tables in the input directory.")
    parser.add_argument("--sample", type=bool, default=False, help="Whether to use sample")
    parser.add_argument("--input_dir", type=str, default="./askem-arpa-h-project/data/extracted-tables", help="The directory containing the tables.")
    parser.add_argument("--output_dir", type=str, default="./askem-arpa-h-project/table-union/embeddings_clean/cand/header", help="The directory to store the embeddings.")
    args = parser.parse_args()
    api_key = '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' #TODO: Replace with your own OpenAI API key
    
    main(api_key, args.input_dir, args.output_dir, args.sample)

# Test
# embedding = pd.read_pickle('./embeddings/target-embeddings/Li_data_in_GDC_format_0.pkl')
# print(embedding)  
# print(f"Shape: {len(embedding)}")  
# print(f"Range: {min(embedding)} to {max(embedding)}") 
                    
