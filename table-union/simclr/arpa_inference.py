import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from model import BarlowTwinsSimCLR
from dataset import PretrainTableDataset

def load_checkpoint(ckpt, hp):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.to(device)
    model.load_state_dict(ckpt['model'])

    return model


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
                # all column vectors in the batch
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

    # def encode_tables(table_names, column_names):
    #     tables = []
    #     for table_name, col_name in zip(table_names, column_names):
    #         table = pd.read_csv(os.path.join(table_path, f"tables/{table_name}.csv"))
    #         if model.hp.single_column:
    #             table = pd.DataFrame(table[col_name])
    #         tables.append(table)
    #     vectors = inference_on_tables(tables, model, unlabeled, batch_size=128)
        
    #     assert all(len(vec) == len(table.columns) for vec, table in zip(vectors, tables))

    #     res = []
    #     for vec, cid in zip(vectors, column_names):
    #         if isinstance(cid, str): # for ARPA test
    #             res.append(vec[-1])
    #         elif cid < len(vec):
    #             res.append(vec[cid])
    #         else:
    #             res.append(vec[-1])
    #     return res
    
    tables = []
    for i, column in enumerate(table.columns):
        curr_table = pd.DataFrame(table[column])
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
        
        result = {
            "Candidate column": l_column_ids[index],
            "Top k columns": top_k_column_names
        }
        
        top_k_results.append(result)

    results_path = f'retrived_top_{k}_results.json'
        
    with open(results_path, 'w') as file:
        json.dump(top_k_results, file, indent=4)

    print(f"Results saved to {results_path}")
    
    # embeddings_directory = './gdc_embeddings'
    # os.makedirs(embeddings_directory, exist_ok=True)
    # for i, embedding in enumerate(r_features):
    #     np.save(os.path.join(embeddings_directory, f'{i}.npy'), embedding)

def run_inference(hp):
    table = pd.read_csv(hp.cand_table)
    print(f"Loaded table from {hp.cand_table}")
    ckpt = torch.load(hp.model_path)
    print(f"Loaded model from {hp.model_path}")
    model = load_checkpoint(ckpt, hp)
    evaluate_arpa_matching(model, table, hp)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="small")
    parser.add_argument("--logdir", type=str, default="../../model/")
    parser.add_argument("--run_id", type=int, default=0)
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
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--cand_table", type=str, default='data/tables/Cao.csv')
    # ../data/extracted-tables/Cao_Clinical_data.csv
    parser.add_argument("--model_path", type=str, default='../../model/arpa/model_10_0.pt')
    parser.add_argument("--use_gdc_embeddings", type=bool, default=True)
    
    hp = parser.parse_args()
    
    run_inference(hp)