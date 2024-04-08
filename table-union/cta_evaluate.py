import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

RAW_PATH = './table-union/cta/raw_types'

gt_all = pd.DataFrame()
gen_all = pd.DataFrame()

for filename in tqdm(os.listdir(RAW_PATH), desc="Matching raw types"):
    if filename.endswith(".csv"):
        col_types = pd.read_csv(os.path.join(RAW_PATH, filename))
        file = filename.split("_")[0]
        try:
            gt = pd.read_csv(f'./data/table-matching-ground-truth/ground-truth/{file}.csv')
        
            gt_results = pd.merge(gt, col_types, how='inner', left_on='original_paper_variable_names', right_on='column_name').fillna('None')
            gen_results = pd.merge(gt, col_types, how='right', left_on='original_paper_variable_names', right_on='column_name').fillna('None')
            
            gt_results.drop(columns=['column_name'], inplace=True)
            gen_results = gen_results[['column_name', 'GDC_format_variable_names','generated_column_type']]
            gen_results.rename(columns={'column_name': 'original_paper_variable_names'}, inplace=True)
            
            gt_results.drop_duplicates(subset=['original_paper_variable_names'], keep='first', inplace=True)
            gen_results.drop_duplicates(subset=['original_paper_variable_names'], keep='first', inplace=True)
            
            # drop case_submitter_id and can be inferred from tobacco_smoking_status
            # gt_results.drop(gt_results[(gt_results['GDC_format_variable_names'] == 'case_submitter_id')].index, inplace=True)
            # gt_results.drop(gt_results[(gt_results['GDC_format_variable_names'] == 'can be inferred from tobacco_smoking_status')].index, inplace=True)
            # gen_results.drop(gen_results[(gen_results['GDC_format_variable_names'] == 'case_submitter_id')].index, inplace=True)
            # gen_results.drop(gen_results[(gen_results['GDC_format_variable_names'] == 'can be inferred from tobacco_smoking_status')].index, inplace=True)
            
            gt_results.to_csv(f'./table-union/cta/gt_results/{file}_gt.csv', index=False)
            gen_results.to_csv(f'./table-union/cta/gen_results/{file}_gen.csv', index=False)
            
            gt_all = pd.concat([gt_all, gt_results])
            gen_all = pd.concat([gen_all, gen_results])
        except:
            print(f'No ground truth for {file}')
            continue

gt_all.drop_duplicates(subset=['original_paper_variable_names'], keep='first', inplace=True)
gen_all.drop_duplicates(subset=['original_paper_variable_names'], keep='first', inplace=True)

gt_all.to_csv('./table-union/cta/all_gt.csv', index=False)
gen_all.to_csv('./table-union/cta/all_gen.csv', index=False)

for filename in os.listdir('./table-union/cta/gt_results'):
    gt = pd.read_csv(f'./table-union/cta/gt_results/{filename}')
    file = filename.split('_')[0]
    gen = pd.read_csv(f'./table-union/cta/gen_results/{file}_gen.csv')
    
    gt.fillna('missing', inplace=True)
    gen.fillna('missing', inplace=True)
    
    gt_micro = precision_score(gt['GDC_format_variable_names'], gt['generated_column_type'], average='micro')
    gt_precision = precision_score(gt['GDC_format_variable_names'], gt['generated_column_type'], average='macro')
    gt_recall = recall_score(gt['GDC_format_variable_names'], gt['generated_column_type'], average='macro')
    gt_f1 = f1_score(gt['GDC_format_variable_names'], gt['generated_column_type'], average='macro')
    
    gen_micro = precision_score(gen['GDC_format_variable_names'], gen['generated_column_type'], average='micro')
    gen_precision = precision_score(gen['GDC_format_variable_names'], gen['generated_column_type'], average='macro')
    gen_recall = recall_score(gen['GDC_format_variable_names'], gen['generated_column_type'], average='macro')
    gen_f1 = f1_score(gen['GDC_format_variable_names'], gen['generated_column_type'], average='macro')
    
    if 'precision' in locals():
        precision = pd.concat([precision, pd.DataFrame([[file.split('_')[0], gt_micro, gt_precision, gt_recall, gt_f1, gen_micro, gen_precision, gen_recall, gen_f1]], columns=['table_name', 'gt_micro_precision', 'gt_precision', 'gt_recall', 'gt_f1', 'gen_micro_precision', 'gen_precision', 'gen_recall', 'gen_f1'])])
    else:
        precision = pd.DataFrame([[file.split('_')[0], gt_micro, gt_precision, gt_recall, gt_f1, gen_micro, gen_precision, gen_recall, gen_f1]], columns=['table_name', 'gt_micro_precision', 'gt_precision', 'gt_recall', 'gt_f1', 'gen_micro_precision', 'gen_precision', 'gen_recall', 'gen_f1'])
        
precision.to_csv('./table-union/cta/result_by_table.csv', index=False)

gt_micro = precision_score(gt_all['GDC_format_variable_names'], gt_all['generated_column_type'], average='micro')
gt_precision = precision_score(gt_all['GDC_format_variable_names'], gt_all['generated_column_type'], average='macro')
gt_recall = recall_score(gt_all['GDC_format_variable_names'], gt_all['generated_column_type'], average='macro')
gt_f1 = f1_score(gt_all['GDC_format_variable_names'], gt_all['generated_column_type'], average='macro')

gen_micro = precision_score(gen_all['GDC_format_variable_names'], gen_all['generated_column_type'], average='micro')
gen_precision = precision_score(gen_all['GDC_format_variable_names'], gen_all['generated_column_type'], average='macro')
gen_recall = recall_score(gen_all['GDC_format_variable_names'], gen_all['generated_column_type'], average='macro')
gen_f1 = f1_score(gen_all['GDC_format_variable_names'], gen_all['generated_column_type'], average='macro')

with open('./table-union/cta/results_all_tables.txt', 'w') as f:
    f.write(f'gt_micro_precision: {gt_micro}\n')
    f.write(f'gt_precision: {gt_precision}\n')
    f.write(f'gt_recall: {gt_recall}\n')
    f.write(f'gt_f1: {gt_f1}\n')
    f.write(f'gen_micro_precision: {gen_micro}\n')
    f.write(f'gen_precision: {gen_precision}\n')
    f.write(f'gen_recall: {gen_recall}\n')
    f.write(f'gen_f1: {gen_f1}\n')
