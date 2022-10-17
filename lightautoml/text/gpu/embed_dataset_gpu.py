"""Pytorch Datasets for text features."""

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
import cupy as cp
import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer
import torch

try:
    from transformers import AutoTokenizer
except:
    import warnings

    warnings.warn("'transformers' - package isn't installed")


class BertDataset_gpu:
    """Dataset class with transformers tokenization."""

    def __init__(self, sentences: Sequence[str], max_length: int, model_name: str, **kwargs: Any):
        """Class for preparing transformers input.

        Args:
            sentences: List of tokenized sentences.
            max_length: Max sentence length.
            model_name: Name of transformer model.

        """
        self.sentences = sentences
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def __getitem__(self, idx: int) -> Dict[str, cp.ndarray]:
        # sent = cudf.Series([self.sentences[idx]])
        sent = self.sentences[idx]
        _split = sent.split("[SEP]")
        sent = _split if len(_split) == 2 else (sent,)
        # sent = cudf.Series(_split) if len(_split) == 2 else cudf.Series([sent])
        data = self.tokenizer.encode_plus(
            *sent, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True
        )
        # data = self.subword(sent, add_special_tokens=True, max_length=self.max_length, max_num_rows=len(sent),
        #                     padding="max_length", truncation=True, return_tensors='cp', return_token_type_ids=False)
        # data.pop("metadata", None)
        # if len(sent) == 2:
        #     data_new = {}
        #     data_new["input_ids"] = cp.zeros(self.max_length)
        #     n1, n2 = cp.sum(data["attention_mask"][0]), cp.sum(data["attention_mask"][1])
        #     data_new["input_ids"][:n1] = data["input_ids"][0][:n1]
        #     data_new["input_ids"][n1:n1+n2-1] = data["input_ids"][1][1:n2]
        #     data_new["attention_mask"] = cp.zeros(self.max_length, dtype=int)
        #     data_new["attention_mask"][:n1+n2-1] = 1
        #     data_new["token_type_ids"] = cp.zeros(self.max_length, dtype=int)
        #     data_new["token_type_ids"][n1:n1+n2-1] = 1
        #     return data_new
        # else:
        #     data["token_type_ids"] = cp.zeros(self.max_length, dtype=int)
        #     return data
        return {i: np.array(data[i]) for i in data.keys()}

    def __len__(self) -> int:
        return len(self.sentences)


class EmbedDataset_gpu:
    """Dataset class for extracting word embeddings."""

    def __init__(self, sentences: Sequence[str], embedding_model: Dict, max_length: int, embed_size: int, **kwargs):
        """Class for transforming list of tokens to dict of embeddings and sentence length.

        Args:
            sentences: List of tokenized sentences.
            embedding_model: word2vec, fasstext, etc.
              Should have dict interface {<word>: <embedding>}.
            max_length: Max sentence length.
            embed_size: Size of embedding.
            **kwargs: Not used.

        """
        self.sentences = sentences.str.split(" ")
        self.embedding_model = embedding_model
        self.max_length = max_length
        self.embed_size = embed_size

    def __getitem__(self, idx: int) -> Dict[str, Union[Sequence, int]]:
        result = torch.zeros((self.max_length, self.embed_size))
        length = 0
        for word in self.sentences[idx]:
            if word in self.embedding_model:
                result[length, :] = self.embedding_model[word]
                length += 1
                if length >= self.max_length:
                    break
        return {"text": result, "length": length if length > 0 else 1}

    def __len__(self) -> int:
        return len(self.sentences)
