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


class BertDatasetGPU:
    """Dataset class with transformers tokenization.

    Class for preparing transformers input.

    Args:
        sentences: List of tokenized sentences.
        max_length: Max sentence length.
        model_name: Name of transformer model.
    """

    def __init__(self, sentences: Sequence[str], max_length: int, model_name: str, **kwargs: Any):
        self.sentences = sentences
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def __getitem__(self, idx: int) -> Dict[str, cp.ndarray]:
        sent = self.sentences[idx]
        _split = sent.split("[SEP]")
        sent = _split if len(_split) == 2 else (sent,)
        data = self.tokenizer.encode_plus(
            *sent, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True
        )
        return {i: np.array(data[i]) for i in data.keys()}

    def __len__(self) -> int:
        return len(self.sentences)


class EmbedDatasetGPU:
    """Dataset class for extracting word embeddings.

    Class for transforming list of tokens to dict of embeddings and sentence length.

    Args:
        sentences: List of tokenized sentences.
        embedding_model: word2vec, fasstext, etc.
          Should have dict interface {<word>: <embedding>}.
        max_length: Max sentence length.
        embed_size: Size of embedding.
        **kwargs: Not used.
    """

    def __init__(self, sentences: Sequence[str], embedding_model: Dict, max_length: int, embed_size: int, **kwargs):
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
