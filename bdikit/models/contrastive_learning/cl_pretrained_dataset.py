import pandas as pd
import torch
from bdikit.models.contrastive_learning.cl_preprocessor import (
    preprocess,
)
from torch.utils import data
from transformers import AutoTokenizer

lm_mp = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}


class PretrainTableDataset(data.Dataset):
    def __init__(self, max_len=128, lm="roberta", sample_meth="head"):
        super().__init__()
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.sample_meth = sample_meth

    def _tokenize(self, table: pd.DataFrame):
        res = []
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
        tfidfDict = None

        column_mp = {}

        for column in table.columns:
            tokens = preprocess(
                table[column], tfidfDict, max_tokens, self.sample_meth
            )  # from preprocessor.py
            col_text = (
                self.tokenizer.cls_token
                + column
                + self.tokenizer.sep_token
                + self.tokenizer.sep_token.join(tokens[:max_tokens])
            )
            column_mp[column] = len(res)
            res += self.tokenizer.encode(
                text=col_text,
                max_length=budget,
                add_special_tokens=False,
                truncation=True,
            )
        return res, column_mp

    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
        """
        x_ori, x_aug, cls_indices = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [
            xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in x_ori
        ]
        x_aug_new = [
            xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in x_aug
        ]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)

        return (
            torch.LongTensor(x_ori_new),
            torch.LongTensor(x_aug_new),
            (cls_ori, cls_aug),
        )
