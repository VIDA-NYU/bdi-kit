import os
import pandas as pd
import json
from scipy.spatial.distance import cosine

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

def calculate_pairwise_similarities(target_path, candidate_path):
    """Calculate pairwise cosine similarities between target and candidate embeddings."""
    target_files = [f for f in os.listdir(target_path) if f.endswith('.pkl')]
    candidate_files = [f for f in os.listdir(candidate_path) if f.endswith('.pkl')]

    # Load embeddings for target and candidate tables
    target_embeddings = {parse_file_name(f): load_embeddings(os.path.join(target_path, f)) for f in target_files}
    candidate_embeddings = {parse_file_name(f): load_embeddings(os.path.join(candidate_path, f)) for f in candidate_files}

    similarities = []
    for (target_table, target_col), target_emb in target_embeddings.items():
        for (candidate_table, candidate_col), candidate_emb in candidate_embeddings.items():
            similarity = 1 - cosine(target_emb, candidate_emb)
            similarities.append((similarity, f"{target_table}_{target_col}", f"{candidate_table}_{candidate_col}"))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities

def map_similarities(similarities_df, table_map):
    """Map similarities to human-readable table and column names."""
    mapped_data = []
    for _, row in similarities_df.iterrows():
        target_table, target_col = map_table_column(row['TargetColumn'], table_map)
        cand_table, cand_col = map_table_column(row['CandidateColumn'], table_map)
        similarity = row['Similarity']
        mapped_data.append({
            'TargetTable': target_table,
            'CandidateTable': cand_table,
            'TargetColumn': target_col,
            'CandidateColumn': cand_col,
            'Similarity': similarity
        })
    return pd.DataFrame(mapped_data)

def main(input_target_dir, input_cand_dir, output_dir, map_columns):
    similarities = calculate_pairwise_similarities(input_target_dir, input_cand_dir)

    # Save all similarities to CSV
    all_similarities_df = pd.DataFrame(similarities, columns=['Similarity', 'TargetColumn', 'CandidateColumn'])
    all_similarities_csv_path = os.path.join(output_dir, 'cand_target.csv')
    all_similarities_df.to_csv(all_similarities_csv_path, index=False)
    print(f"Saved all similarities to {all_similarities_csv_path}.")

    if map_columns:
        with open('./askem-arpa-h-project/table-union/table_map.json', 'r') as json_file:
            table_map = json.load(json_file)
        similarities_df = pd.read_csv(all_similarities_csv_path)
        mapped_df = map_similarities(similarities_df, table_map)
        mapped_csv_path = os.path.join(output_dir, 'cand_target_remapped.csv')
        mapped_df.to_csv(mapped_csv_path, index=False)
        print(f"Saved mapped similarities to {mapped_csv_path}.")
    else:
        print("Mapping not requested. Process completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_target_dir', type=str, default="./askem-arpa-h-project/table-union/embeddings/target/header/", help='Input directory path of target embeddings')
    parser.add_argument('--input_cand_dir', type=str, default="./askem-arpa-h-project/table-union/embeddings/cand/header/", help='Input directory path of candidate embeddings')
    parser.add_argument('--output_dir', type=str, default="./askem-arpa-h-project/table-union/results", help='Output directory path')
    parser.add_argument('--map', dest='map_columns', action='store_true', help='Map columns to readable names')
    parser.set_defaults(map_columns=True)
    args = parser.parse_args()
    
    main(args.input_target_dir, args.input_cand_dir, args.output_dir, args.map_columns)
