import pandas as pd

def load_and_prepare_data(ground_truth_path, remapped_path):
    """
    Loads and prepares the ground truth and candidate target remapped dataframes.
    """
    # Load dataframes
    ground_truth_df = pd.read_csv(ground_truth_path)
    cand_target_remapped_df = pd.read_csv(remapped_path)
    
    return ground_truth_df, cand_target_remapped_df

def find_matches(ground_truth_df, cand_target_remapped_df):
    """
    Finds matches between the ground truth and the candidate target remapped dataframes.
    """ 
    ground_truth_dict = {row['candidate']: row['target'] for _, row in ground_truth_df.iterrows()}
    results = []
    
    for match_index, match_row in cand_target_remapped_df.iterrows():
        candidate_column = match_row['CandidateColumn']
        target_column = match_row['TargetColumn']

        if candidate_column in ground_truth_dict \
            and ground_truth_dict[candidate_column] == target_column\
                and not any(result['Candidate'] == candidate_column for result in results):
            results.append({
                'Candidate': candidate_column,
                'Target': target_column,
                'Similarity': match_row['Similarity'],
                'Index': match_index
            })
                
    return results

def main():
    ground_truth_path = './askem-arpa-h-project/data/table-matching-ground-truth/Dou.csv'
    cand_target_remapped_path = './askem-arpa-h-project/table-union/results/cand_target_remapped.csv'
    output_path = './askem-arpa-h-project/table-union/results/match.csv'
    
    ground_truth_df, cand_target_remapped_df = load_and_prepare_data(ground_truth_path, cand_target_remapped_path)
    results = find_matches(ground_truth_df, cand_target_remapped_df)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    main()
