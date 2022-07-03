"""Pytorch Datasets for text features."""

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np


try:
    from transformers import AutoTokenizer
except:
    import warnings

    warnings.warn("'transformers' - package isn't installed")


class BertDataset:
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

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sent = self.sentences[idx]
        _split = sent.split("[SEP]")
        sent = _split if len(_split) == 2 else (sent,)
        data = self.tokenizer.encode_plus(
            *sent, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True
        )
        return {i: np.array(data[i]) for i in data.keys()}

    def __len__(self) -> int:
        return len(self.sentences)


class EmbedDataset:
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
        self.sentences = sentences
        self.embedding_model = embedding_model
        self.max_length = max_length
        self.embed_size = embed_size

    def __getitem__(self, idx: int) -> Dict[str, Union[Sequence, int]]:
        result = np.zeros((self.max_length, self.embed_size))
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
