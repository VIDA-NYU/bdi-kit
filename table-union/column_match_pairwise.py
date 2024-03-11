import os
import pandas as pd
import json
from scipy.spatial.distance import cosine
from itertools import combinations

def load_embeddings(file_path):
    """Load embeddings from a pickle file."""
    embeddings_df = pd.read_pickle(file_path)
    return embeddings_df.values.ravel()

def parse_file_name(file_name):
    """Parse the table name and column index from a file name."""
    parts = file_name.rsplit('_', 1)
    table_name = parts[0]
    column_index = parts[1].split('.')[0]
    return table_name, int(column_index)

def map_table_column(identifier, table_map):
    """Map table and column identifiers to original column names."""
    table_name, col_index = identifier.rsplit('_', 1)
    col_name = table_map[table_name].get(col_index, f"Column{col_index}")
    return table_name, col_name

def calculate_pairwise_similarities(embeddings_path):
    """Calculate pairwise cosine similarities between embeddings."""
    embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('.pkl')]
    embeddings_dict = {}
    for file in embedding_files:
        file_path = os.path.join(embeddings_path, file)
        table_name, column_index = parse_file_name(file)
        embeddings_dict[(table_name, column_index)] = load_embeddings(file_path)

    similarities = []
    for (table1, col1), emb1 in embeddings_dict.items():
        for (table2, col2), emb2 in embeddings_dict.items():
            if table1 != table2: 
                similarity = 1 - cosine(emb1, emb2)
                similarities.append((similarity, f"{table1}_{col1}", f"{table2}_{col2}"))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities

def map_similarities(similarities_df, table_map):
    """Map similarities to human-readable table and column names."""
    mapped_data = []
    for _, row in similarities_df.iterrows():
        table1, column1 = map_table_column(row['Column1'], table_map)
        table2, column2 = map_table_column(row['Column2'], table_map)
        similarity = row['Similarity']
        mapped_data.append({
            'Table1': table1,
            'Table2': table2,
            'Column1': column1,
            'Column2': column2,
            'Similarity': similarity
        })
    return pd.DataFrame(mapped_data)

def main(input_dir, output_dir, map_columns):
    embeddings_path = input_dir
    similarities = calculate_pairwise_similarities(embeddings_path)

    # Save all similarities to CSV
    all_similarities_df = pd.DataFrame(similarities, columns=['Similarity', 'Column1', 'Column2'])
    all_similarities_csv_path = os.path.join(output_dir, 'all_pair.csv')
    all_similarities_df.to_csv(all_similarities_csv_path, index=False)
    print(f"Saved all similarities to {all_similarities_csv_path}.")

    if map_columns:
        with open('./askem-arpa-h-project/table-union/table_map.json', 'r') as json_file:
            table_map = json.load(json_file)

        similarities_df = pd.read_csv(all_similarities_csv_path)
        mapped_df = map_similarities(similarities_df, table_map)
        mapped_csv_path = os.path.join(output_dir, 'all_pair_mapped.csv')
        mapped_df.to_csv(mapped_csv_path, index=False)
        print(f"Saved mapped similarities to {mapped_csv_path}.")
    else:
        print("Mapping not requested. Process completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', type=str, default="./askem-arpa-h-project/table-union/embeddings/cand/header/", help='Input directory path')
    parser.add_argument('--output_dir', type=str, default="./results", help='Output directory path')
    parser.add_argument('--map', dest='map_columns', action='store_true', help='Map columns to readable names')
    parser.set_defaults(map_columns=True)
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.map_columns)
