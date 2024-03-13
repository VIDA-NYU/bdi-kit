import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

matches_df_raw = pd.read_csv('./askem-arpa-h-project/table-union/results/clean_table_level_match/Dou_match_mapped.csv')
ground_truth_df_raw = pd.read_csv('./askem-arpa-h-project/data/table-matching-ground-truth/DouManual.csv')
incorrect_matches = matches_df_raw[~matches_df_raw['TargetColumn'].isin(ground_truth_df_raw['target'])]

matches_df = matches_df_raw.sort_values(by=['TargetColumn'])
ground_truth_df = ground_truth_df_raw.sort_values(by=['target'])

matches_df = matches_df[matches_df['TargetColumn'].isin(ground_truth_df['target'])]

# join the two dataframes on the target column
result_df = pd.merge(ground_truth_df, matches_df, left_on='target', right_on='TargetColumn', how='left')
result_df = result_df[['target', 'candidate', 'CandidateColumn', 'Similarity']]
# result_df = pd.concat([result_df, incorrect_matches.rename(columns={'TargetColumn': 'target'}).assign(candidate='N/A')], ignore_index=True)
result_df = pd.concat(
    [result_df.drop(columns=['TargetTable', 'CandidateTable'], errors='ignore').dropna(subset=['candidate']), 
    incorrect_matches.rename(columns={'TargetColumn': 'target'}).drop(columns=['TargetTable', 'CandidateTable'], errors='ignore').assign(candidate='N/A')],
    ignore_index=True
)
result_df['Similarity'] = result_df['Similarity'].round(2)
result_df = result_df.sort_values(by=['Similarity'], ascending=False)

result_df.to_csv('./askem-arpa-h-project/table-union/results/Duo_clean_result.csv', index=False)

true_positives = matches_df[matches_df.apply(
    lambda x: (x['TargetColumn'], x['CandidateColumn']) in zip(
        ground_truth_df['target'], ground_truth_df['candidate']), axis=1)]

false_positives = matches_df[~matches_df.apply(
    lambda x: (x['TargetColumn'], x['CandidateColumn']) in zip(
        ground_truth_df['target'], ground_truth_df['candidate']), axis=1)]

false_negatives = ground_truth_df[~ground_truth_df.apply(
    lambda x: (x['target'], x['candidate']) in zip(
        matches_df['TargetColumn'], matches_df['CandidateColumn']), axis=1)]

precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
