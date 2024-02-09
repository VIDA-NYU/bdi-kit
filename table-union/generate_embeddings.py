import os
import pandas as pd
from openai import OpenAI

class CSVEmbeddingsGenerator:
    def __init__(self, api_key, directory, output_directory, model="text-embedding-3-small"):
        self.api_key = api_key
        self.directory = directory
        self.output_directory = output_directory
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        os.makedirs(self.output_directory, exist_ok=True)

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding


    def generate_embeddings(self, engine="text-similarity-babbage-001"):
        for filename in os.listdir(self.directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.directory, filename)
                df = pd.read_csv(file_path)
                
                # Process each column in the DataFrame
                for col_index, col_name in enumerate(df.columns):
                    # Select 5 random rows from the column
                    if len(df[col_name].dropna()) < 5:
                        print(f"Not enough data in {col_name} to generate embeddings.")
                        continue
                    
                    random_rows = df[col_name].dropna().sample(5).tolist()
                    serialized_input = f"{col_name}: {', '.join([str(row) for row in random_rows])}"
                    
                    # Generate the embedding for the serialized input
                    embedding = self.get_embedding(serialized_input)
                    
                    # Store the embeddings
                    self._store_embeddings(filename, col_index, embedding)
                    
                    print(f"Generated embeddings for {filename}, column {col_name}.")

    def _store_embeddings(self, filename, col_index, embeddings):
        base_filename = os.path.splitext(filename)[0]
        embeddings_file_path = os.path.join(self.output_directory, f"{base_filename}_{col_index}.pkl")
        
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.to_pickle(embeddings_file_path)

api_key = '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' #TODO: Replace with your own OpenAI API key
input_directory = './Starter-Kit' 
output_directory = './embeddings/target-embeddings' 

embedder = CSVEmbeddingsGenerator(api_key, input_directory, output_directory)
embedder.generate_embeddings()

# Test
# embedding = pd.read_pickle('./embeddings/target-embeddings/Li_data_in_GDC_format_0.pkl')
# print(embedding)  
# print(f"Shape: {len(embedding)}")  
# print(f"Range: {min(embedding)} to {max(embedding)}") 
                    
