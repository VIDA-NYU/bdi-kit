import torch
import numpy as np
import pandas as pd
import os
import json
import argparse
import tiktoken
from tqdm import tqdm
from torch.utils import data
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from model import BarlowTwinsSimCLR
from dataset import PretrainTableDataset
from cta import CTA

def load_checkpoint(ckpt, hp):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.to(device)
    model.load_state_dict(ckpt['model'])

    return model

def sample_to_15_rows(table):
    """
    Guarantee that the table has 15 rows
    """
    if len(table) > 15:
        unique_rows = table.drop_duplicates()
        num_unique_rows = len(unique_rows)
        if num_unique_rows <= 15:
            needed_rows = 15 - num_unique_rows
            additional_rows = table[~table.index.isin(unique_rows.index)].sample(n=needed_rows, replace=True, random_state=1)
            table = pd.concat([unique_rows, additional_rows])
        else:
            table = unique_rows.sample(n=15, random_state=1)
    return table

def inference_on_tables(tables: List[pd.DataFrame],
                        model: BarlowTwinsSimCLR,
                        unlabeled: PretrainTableDataset,
                        batch_size=128,
                        total=None):
    
    total=total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        x, _ = unlabeled._tokenize(table)

        batch.append((x, x, []))
        if tid == total - 1 or len(batch) == batch_size:
            # model inference
            with torch.no_grad():
                x, _, _ = unlabeled.pad(batch)
                column_vectors = model.inference(x)
                ptr = 0
                for xi in x:
                    current = []
                    for token_id in xi:
                        if token_id == unlabeled.tokenizer.cls_token_id:
                            current.append(column_vectors[ptr].cpu().numpy())
                            ptr += 1
                    results.append(current)
            batch.clear()

    return results


def evaluate_arpa_matching(model: BarlowTwinsSimCLR,
                           table: pd.DataFrame,
                           hp):
    table_path = 'data/tables'
    k = hp.top_k
    results = []
    
    unlabeled = PretrainTableDataset.from_hp(table_path, hp)
    
    tables = []
    for i, column in enumerate(table.columns):
        curr_table = pd.DataFrame(table[column])
        curr_table = sample_to_15_rows(curr_table)
        tables.append(curr_table)
    vectors = inference_on_tables(tables, model, unlabeled, batch_size=128)
    l_features = [vec[-1] for vec in vectors]
    print(f"Table features extracted from {len(table.columns)} columns")
    
    gdc_ds = pd.read_csv(os.path.join(table_path, 'gdc_table.csv'))
    if hp.use_gdc_embeddings:
        r_features = []
        embeddings_directory = './gdc_embeddings'
        for i in range(len(gdc_ds.columns)):
            embedding = np.load(os.path.join(embeddings_directory, f'{i}.npy'))
            r_features.append(embedding)
        print(f"Loaded GDC embeddings from {embeddings_directory}")
    else:
        gdc_tables = []
        for i, column in enumerate(gdc_ds.columns):
            curr_table = pd.DataFrame(gdc_ds[column])
            curr_table = sample_to_15_rows(curr_table)
            gdc_tables.append(curr_table)
        gdc_vectors = inference_on_tables(gdc_tables, model, unlabeled, batch_size=128)
        r_features = [vec[-1] for vec in gdc_vectors]
        print(f"GDC table features extracted from {len(gdc_ds.columns)} columns")
    
    cosine_sim = cosine_similarity(l_features, r_features)
    print(f"Computed cosine similarity matrix")
    
    top_k_results = []
    l_column_ids = table.columns
    gt_column_ids = gdc_ds.columns
    
    for index, similarities in enumerate(cosine_sim):
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_column_names = [gt_column_ids[i] for i in top_k_indices]
        top_k_similarities = [str(round(similarities[i], 4)) for i in top_k_indices]
        top_k_columns = list(zip(top_k_column_names, top_k_similarities))
        
        result = {
            "Candidate column": l_column_ids[index],
            "Top k columns": top_k_columns
        }
        top_k_results.append(result)

    table_name = hp.cand_table.split('/')[-1].split('.')[0]
    results_path = f'./arpa_result/top_{k}_{table_name}.json'
    with open(results_path, 'w') as file:
        json.dump(top_k_results, file, indent=4)
    print(f"Top {k} results saved to {results_path}")
    
    if hp.save_embeddings:
        embeddings_directory = './gdc_embeddings'
        os.makedirs(embeddings_directory, exist_ok=True)
        for i, embedding in enumerate(r_features):
            np.save(os.path.join(embeddings_directory, f'{i}.npy'), embedding)
    
    return top_k_results
        

def gpt_cta(top_k_results, table, hp):
    api_key = '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'
    annotator = CTA(api_key)
    m = hp.cta_m
    results = []
    for result in tqdm(top_k_results):
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
            "Candidate column": candidate_column,
            "Target GDC variable name(s)": col_type
        }
        results.append(match)

    table_name = hp.cand_table.split('/')[-1].split('.')[0]
    results_path = f'./arpa_result/top_{m}_{table_name}.json'
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Top-{m} results saved to {results_path}")


def run_inference(hp):
    table = pd.read_csv(hp.cand_table)
    print(f"Loaded table from {hp.cand_table}")
    if hp.top_k in [10, 20, 50]:
        model_idx = hp.top_k
    else:
        model_idx = 20
    model_path = os.path.join(hp.logdir, hp.task, f"model_{model_idx}_{hp.run_id}.pt")
    ckpt = torch.load(model_path)
    print(f"Loaded model from {model_path}")
    model = load_checkpoint(ckpt, hp)
    top_k_results = evaluate_arpa_matching(model, table, hp)
    if hp.use_cta:
        print(f"Running CTA on top-{hp.cta_m} results...")
        gpt_cta(top_k_results, table, hp)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="arpa")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='drop_col,sample_row')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--sample_meth", type=str, default='head')
    parser.add_argument("--gpt", type=bool, default=True)
    
    # Path to models directory (https://drive.google.com/drive/folders/1ntaEq4MDcEJ8gsZLU_uS-oosOnEmS12r?usp=sharing)
    parser.add_argument("--logdir", type=str, default="../../models")
    # Run ID
    parser.add_argument("--run_id", type=int, default=1)
    # Path to the candidate table
    parser.add_argument("--cand_table", type=str, default='../data/extracted-tables/Dou_UCEC_CPTAC3_meta_table_V2.1.csv')
    # Top k results to return by cosine similarity
    parser.add_argument("--top_k", type=int, default=20)
    # Save the GDC embeddings
    parser.add_argument("--save_embeddings", dest="save_embeddings", action="store_true")
    # Use stored GDC embeddings
    parser.add_argument("--use_gdc_embeddings", dest="use_gdc_embeddings", action="store_true")
    # Use CTA to further refine the results to top 1-5
    parser.add_argument("--use_cta", dest="use_cta", action="store_true")
    # Set the number of top results to run CTA on
    parser.add_argument("--cta_m", type=int, default=1)
    # Output directory for the results
    parser.add_argument("--output_dir", type=str, default='./arpa_result')
    
    hp = parser.parse_args()
    if not os.path.exists(hp.output_dir):
        os.makedirs(hp.output_dir)
    run_inference(hp)