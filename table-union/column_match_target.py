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

def calculate_similarities(target_path, candidate_path):
    """Calculate pairwise cosine similarities between target and candidate embeddings."""
    target_files = [f for f in os.listdir(target_path) if f.endswith('.pkl')]
    candidate_files = [f for f in os.listdir(candidate_path) if f.endswith('.pkl')]

    target_embeddings = {parse_file_name(f): load_embeddings(os.path.join(target_path, f)) for f in target_files}
    candidate_embeddings = {parse_file_name(f): load_embeddings(os.path.join(candidate_path, f)) for f in candidate_files}
    gt_table_names = [candidate_table.split('_')[0] for (candidate_table, _), _ in candidate_embeddings.items()]
    gt_table_names = list(set(gt_table_names))
    similarities = {gt_table_name: [] for gt_table_name in gt_table_names}
    for (candidate_table, candidate_col), candidate_emb in candidate_embeddings.items():
        for (target_table, target_col), target_emb in target_embeddings.items():
            similarity = 1 - cosine(target_emb, candidate_emb)
            similarities[f"{candidate_table.split('_')[0]}"].append((similarity, f"{target_table}_{target_col}", f"{candidate_table}_{candidate_col}"))
    
    for candidate_table, sim_list in similarities.items():
        sim_list.sort(reverse=True, key=lambda x: x[0])
    return similarities

def generate_top_matches(similarities):
    """Generate top matches from the similarities."""
    return 0

def map_similarities(sim_df, table_map):
    """Map similarities to human-readable table and column names."""
    mapped_data = []
    for _, row in sim_df.iterrows():
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
    similarities = calculate_similarities(input_target_dir, input_cand_dir)
    for candidate_table, sim_list in similarities.items():
        sim_df = pd.DataFrame(sim_list, columns=['Similarity', 'TargetColumn', 'CandidateColumn'])
        sim_df = sim_df.groupby('TargetColumn').first().reset_index()
        sim_csv_path = os.path.join(output_dir, f'{candidate_table}_match.csv')
        sim_df.to_csv(sim_csv_path, index=False)

        if map_columns:
            with open('./askem-arpa-h-project/table-union/table_map.json', 'r') as json_file:
                table_map = json.load(json_file)
            mapped_df = map_similarities(sim_df, table_map)
            mapped_csv_path = os.path.join(output_dir, f'{candidate_table}_match_mapped.csv')
            mapped_df.to_csv(mapped_csv_path, index=False)
            print(f"Mapped similarities saved to {mapped_csv_path}.")
        else:
            print("Mapping not requested. Process completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_target_dir', type=str, default="./askem-arpa-h-project/table-union/embeddings/target/header_sample_5/", help='Input directory path of target embeddings')
    parser.add_argument('--input_cand_dir', type=str, default="./askem-arpa-h-project/table-union/embeddings/cand/header_sample_5/", help='Input directory path of candidate embeddings')
    parser.add_argument('--output_dir', type=str, default="./askem-arpa-h-project/table-union/results/table_level_match_sample", help='Output directory path')
    parser.add_argument('--map', dest='map_columns', action='store_true', help='Map columns to readable names')
    parser.set_defaults(map_columns=True)
    args = parser.parse_args()
    
    main(args.input_target_dir, args.input_cand_dir, args.output_dir, args.map_columns)
