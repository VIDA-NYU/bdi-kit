from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import List, Dict, NamedTuple, Optional
import torch
import numpy as np
import pandas as pd
from bdikit.models import ColumnEmbedder


class ValueToken(NamedTuple):
    text: str
    tokens: List[int]


class EmbeddingResult(NamedTuple):
    column: np.ndarray
    values: Optional[Dict[str, np.ndarray]]


class SpladeEmbedder(ColumnEmbedder):

    def __init__(self, model_id: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        # extract the ID position to text token mappings
        self.idx2token = {
            idx: token for token, idx in self.tokenizer.get_vocab().items()
        }

    def get_embeddings(self, table: pd.DataFrame) -> List[np.ndarray]:
        print(f"Computing SPLADE vectors for {len(table.columns)} columns...")
        vectors: List[np.ndarray] = []
        for column in table.columns:
            if pd.api.types.is_string_dtype(table[column]):
                uniq_values = table[column].unique().tolist()
            else:
                uniq_values = table[column].unique().astype(str).tolist()

            embs = self.embed_values(uniq_values, embed_values=False)
            vectors.append(embs.column)
        print(f"done.")
        return vectors

    def embed_values(
        self, values: List[str], embed_values: bool = True
    ) -> EmbeddingResult:
        batches = self.split_batches(values)
        embedings = [self.embed_values_batch(batch, embed_values) for batch in batches]

        if len(embedings) > 1:
            column_embedding = np.mean([emb.column for emb in embedings], axis=0)
        else:
            column_embedding = embedings[0].column

        if embed_values:
            value_embeddings = {}
            for emb in embedings:
                assert (
                    emb.values is not None
                ), "Embeddings for individual values are missing"
                for value, value_emb in emb.values.items():
                    assert (
                        value not in value_embeddings
                    ), f"Value {value} already exists in embeddings"
                    value_embeddings[value] = value_emb
        else:
            value_embeddings = None

        return EmbeddingResult(column=column_embedding, values=value_embeddings)

    def embed_values_batch(
        self, values: List[ValueToken], embed_values: bool = True
    ) -> EmbeddingResult:
        # serialize the values into a single string
        text = "[SEP]".join([v.text for v in values])

        text_tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        model_output = self.model(**text_tokens)
        text_token_ids = text_tokens["input_ids"].squeeze()

        values_vectors = None
        if embed_values:
            values_vectors = {}
            for value in values:
                # [1:] removes the first ([CLS]) token that is added automatically
                # by the tokenizer. For instance, if "my text" is tokenized as
                # ["[CLS]", "my", "text", "[SEP]"], we keep only token ids for
                # ["my", "text", "[SEP]"]. This is because the [CLS] token is not
                # present in the text string that is fed to the model above.
                value_token_ids = value.tokens[1:]

                # find the indexes in the context indexes that correspond to the current value
                first_token_indexes = torch.where(text_token_ids == value_token_ids[0])[
                    0
                ].tolist()

                token_indexes = []
                for idx in first_token_indexes:
                    indexes = [i for i in range(len(value_token_ids))]
                    if all(
                        text_token_ids[idx + i] == value_token_ids[i] for i in indexes
                    ):
                        token_indexes = [int(idx + i) for i in indexes]
                        break
                assert len(token_indexes) > 0, "Failed to find all token indexes"

                # filter out the token indexes that do not correspond to the current value
                in_context_logits = model_output.logits[0][token_indexes]
                attention_masks = text_tokens.attention_mask[0][token_indexes]

                # compute SPLADE importance vector based on the logits/masks of the current value
                importance_vector = (
                    torch.max(
                        torch.log(1 + torch.relu(in_context_logits))
                        * attention_masks.unsqueeze(-1),
                        dim=0,
                    )[0]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                assert (
                    value.text not in values_vectors
                ), f"Value {value.text} already exists in embeddings"

                values_vectors[value.text] = importance_vector

        # compute SLADE for all values
        importance_vector = (
            torch.max(
                torch.log(1 + torch.relu(model_output.logits))
                * text_tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        return EmbeddingResult(column=importance_vector, values=values_vectors)

    def split_batches(self, values: List[str]) -> List[List[ValueToken]]:
        max_length: int = self.tokenizer.model_max_length
        assert max_length == 512, f"The model max length is {max_length}, expected 512"

        tokenized_values = self.tokenizer(values)["input_ids"]
        batches = []
        curr_batch = []
        total = 0
        for i, token_list in enumerate(tokenized_values):  #  type: ignore
            list_size = len(token_list)
            if total + list_size > max_length:
                batches.append(curr_batch)
                curr_batch = []
                total = 0

            curr_batch.append(ValueToken(values[i], token_list))
            total += list_size

        if len(curr_batch) > 0:
            batches.append(curr_batch)

        return batches

    def to_sparse_vector(
        self, dense_vector: np.ndarray, human_readable: bool = False
    ) -> Dict:
        # find non-zero positions in the importance_vector and extract the non-zero weights
        non_zero_idxs = np.nonzero(dense_vector)[0]
        non_zero_weights = dense_vector[non_zero_idxs]

        output: Dict = {
            "indexes": non_zero_idxs,
            "weights": non_zero_weights,
        }

        if human_readable:
            sparse_dict_tokens = self.readable_tokens_map(
                non_zero_idxs, non_zero_weights
            )
            output["readable"] = sparse_dict_tokens

        return output

    def readable_tokens_map(self, cols: np.ndarray, weights: np.ndarray):
        # map token IDs to human-readable tokens
        sparse_dict_tokens = {
            self.idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }

        # sort so we can see most relevant tokens first
        sparse_dict_tokens = {
            k: v
            for k, v in sorted(
                sparse_dict_tokens.items(), key=lambda item: item[1], reverse=True
            )
        }
        return sparse_dict_tokens
