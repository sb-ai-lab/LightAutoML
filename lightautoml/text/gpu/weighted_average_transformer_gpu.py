"""Weighted average transformer for sequence embeddings."""

import logging
import time

from collections import Counter
from itertools import repeat
from typing import Any
from typing import Dict
from typing import Sequence

import numpy as np
import cupy as cp
import torch

from cuml.feature_extraction.text import CountVectorizer
from tqdm import tqdm


logger = logging.getLogger(__name__)


class WeightedAverageTransformer_gpu:
    """Weighted average of word embeddings.

    Calculate sentence embedding as weighted average of word embeddings.

    Args:
        embedding_model: word2vec, fasstext, etc.
          Should have dict interface {<word>: <embedding>}.
        embed_size: Size of embedding.
        weight_type: 'idf' for idf weights, 'sif' for
          smoothed inverse frequency weights, '1' for all weights are equal.
        use_svd: Substract projection onto first singular vector.
        alpha: Param for sif weights.
        verbose: Add prints.
        **kwargs: Unused arguments.
    """

    name = "WAT_gpu"

    def __init__(
        self,
        embedding_model: Dict,
        embed_size: int,
        weight_type: str = "idf",
        use_svd: bool = True,
        alpha: int = 0.001,
        verbose: bool = False,
        **kwargs: Any,
    ):
        super(WeightedAverageTransformer_gpu, self).__init__()

        if weight_type not in ["sif", "idf", "1"]:
            raise Exception("weights should be one of ['sif', 'idf', '1']")

        self.weight_type = weight_type
        self.alpha = alpha
        self.embedding_model = embedding_model
        self.embed_size = embed_size
        self.use_svd = use_svd
        self.verbose = verbose
        self.weights_ = None
        self.u_ = None
        self.w_all = 0
        self.w_emb = 0

    def get_name(self) -> str:
        """Module name.

        Returns:
            string with module name.

        """
        return self.name + "_" + self.weight_type

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.embed_size

    def reset_statistic(self):
        """Reset module statistics."""

        self.w_all = 0
        self.w_emb = 0

    def get_statistic(self):
        """Get module statistics."""

        logger.info3(f"N_words: {self.w_all}, N_emb: {self.w_emb}, coverage: {self.w_emb / self.w_all}.")

    def get_embedding_(self, sentences: Sequence[str]) -> cp.ndarray:
        n_sent = sentences.shape[0]
        result = torch.zeros((n_sent, self.embed_size))

        for idx in range(n_sent):
            sent = sentences[idx]
            for word in sent:
                self.w_all += 1
                if word in self.embedding_model and word in self.weights_:
                    self.w_emb += 1
                    result[idx] += self.embedding_model[word] * self.weights_[word]

            if len(sent) > 0:
                result[idx] = result[idx] / len(sent)

        return cp.asarray(result)

    def fit(self, sentences: Sequence[str]):
        self.reset_statistic()
        count_vect = CountVectorizer()
        occurrences = count_vect.fit_transform(sentences).get()
        vocab = count_vect.vocabulary_.to_pandas()

        if self.weight_type == "idf":
            nd = np.asarray((occurrences > 0).sum(axis=0)).ravel()
            idf = np.log1p((occurrences.shape[0] + 1) / (nd + 1))
            self.weights_ = {vocab[i]: idf[i] for i in range(idf.shape[0])}

        elif self.weight_type == "sif":
            nd = np.asarray((occurrences > 0).sum(axis=0)).ravel()
            pw = (nd + 1) / (occurrences.shape[0] + 1)
            pw = self.alpha / (self.alpha + pw)
            self.weights_ = {vocab[i]: pw[i] for i in range(pw.shape[0])}

        else:
            self.weights_ = {vocab[i]: 1 for i in range(vocab.shape[0])}
        if self.use_svd:
            if self.verbose:
                sentences = tqdm(sentences)
            res = sentences.str.split(" ").to_pandas()
            sentence_embeddings = self.get_embedding_(res)
            u, _, _ = cp.linalg.svd(sentence_embeddings.T, full_matrices=False)
            self.u_ = u[:, 0]
        return self

    def transform(self, sentences: Sequence[str]) -> cp.ndarray:
        self.reset_statistic()
        res = sentences.str.split(" ").to_pandas()
        sentence_embeddings = self.get_embedding_(res)

        if self.use_svd:
            proj = (self.u_.reshape(-1, 1) * self.u_.dot(sentence_embeddings.T).reshape(1, -1)).T
            sentence_embeddings = sentence_embeddings - proj

        return sentence_embeddings.astype(cp.float32)

    def fit_transform(self, sentences: Sequence[str]) -> cp.ndarray:
        self.fit(sentences)
        return self.transform(sentences)

