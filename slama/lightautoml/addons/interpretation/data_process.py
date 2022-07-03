from collections import Counter
from collections import defaultdict
from random import shuffle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler


class LengthDataset(Dataset):
    """Default dict like PyTorch dataset, with additional info about lenght of sequence."""

    def __init__(self, tokens: List[List[int]], targets: np.ndarray):
        """

        Args:
            tokens: List of tokens number, numbering from zero to dictionary size.
            targets: Array with targets.

        """
        self.tokens = tokens
        self.targets = targets

    def __getitem__(self, idx):
        return {
            "text": self.tokens[idx].clone().detach(),
            "len": len(self.tokens[idx]),
            "target": torch.from_numpy(self.targets[idx]),
        }

    def __len__(self):
        return len(self.targets)


class BySequenceLengthSampler(Sampler):
    """PyTorch sampler with binning by sequnce length."""

    def __init__(
        self,
        data_source: LengthDataset,
        bucket_boundaries: List[int],
        batch_size: int = 64,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        """

        Args:
            data_source: Dataset in dict like format, with key `'len'`.
                The values corresponding to this key should contain
                the length of the sentence (i.e. number of tokens).
            bucket_boundaries: Binning boundaries. The
            batch_size: Number of elements from bin used for batching.
            drop_last: Set to `True` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size,
                else leave the remaining part.
            shuffle: Shuffle flag. If `True` the internally
                and externally bins will be shuffled.

        """
        self.data_source = data_source

        ind_n_len = []

        for i, d in enumerate(self.data_source):
            ind_n_len.append((i, d["len"]))

        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ind_n_len = ind_n_len
        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)
        self.shuffle = shuffle

    def __iter__(self):
        data_buckets = defaultdict(list)
        for i, slen in self.ind_n_len:
            pid = self.to_bucket(slen)
            data_buckets[pid].append(i)

        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        if self.shuffle:
            shuffle(iter_list)

        for i in iter_list:
            yield i.numpy().tolist()

    def __len__(self):
        return len(self.data_source) // self.batch_size + 1

    def to_bucket(self, seq_length):
        valid_buckets = (seq_length >= self.buckets_min) * (seq_length < self.buckets_max)
        bucket_id = torch.nonzero(valid_buckets, as_tuple=True)[0].item()

        return bucket_id

    def shuffle_tensor(self, t):
        if self.shuffle:
            return t[torch.randperm(len(t))]
        else:
            return t


def pad_max_len(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function, that pads sequences to maximum length of text in batch.

    Args:
        batch: List of dictionaries from LengthDataset.

    Returns:
        Dict of vectors of padded texts, targets.

    """
    texts = []
    lens = []
    targets = []
    for d in batch:
        texts.append(d["text"])
        lens.append(d["len"])
        targets.append(d["target"])

    return {
        "text": pad_sequence(texts, batch_first=True, padding_value=0),
        "len": lens,
        "target": torch.stack(targets),
    }


def get_tokenized(data: pd.DataFrame, tokenized_col: str, tokenizer: Callable) -> List[List[str]]:
    """
    Get tokenized column.

    Args:
        data: Dataset with text column.
        tokenized_col: Column used to tokenized.
        tokenizer: Tokenizer function str->List[str].

    Returns:
        Tokenized column.

    """
    tokenized = []
    for sent in data[tokenized_col]:
        tokenized.append(tokenizer(sent.lower()))

    return tokenized


def get_vocab(tokenized: List[List[str]], max_len_vocab: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get vocabulary dataset.

    Args:
        tokenized: Tokenized strings.
        vocab_params: Vocabulary parameters.

    Returns:
        Word to index, index to word.

    """
    counter = Counter()
    for sent in tokenized:
        counter.update(sent)
    max_len_vocab = len(counter) if max_len_vocab == -1 else max_len_vocab
    max_len_vocab = min(max_len_vocab, len(counter))
    word_to_id = {"<PAD>": 0, "<START>": 1, "<UNK>": 2}
    word_to_id.update({v: i + 3 for i, (v, _) in enumerate(counter.most_common(max_len_vocab))})
    id_to_word = {w: k for k, w in word_to_id.items()}

    return word_to_id, id_to_word


def get_embedding_matrix(id_to_word: Dict[int, str], embedder: Any, embedder_dim: int) -> torch.FloatTensor:
    """Create embedding matrix from lookup table and (optionaly) embedder.

    Args:
        id_to_word: Dictionary token-id to word.
        embedder: Lookup table with embeddings.
        embedder_dim: Dimetion of embeddings.

    Returns:
        Weight matrix of embedding layer.

    """
    weights_matrix = np.random.normal(scale=0.6, size=(len(id_to_word), embedder_dim))
    if embedder is not None:
        for i in range(len(id_to_word)):
            word = id_to_word[i]
            try:
                emb = embedder[word]
            except KeyError or TypeError:
                continue
            if sum(emb) != 0:
                weights_matrix[i] = emb

    weights_matrix = torch.from_numpy(weights_matrix)

    return weights_matrix


def map_tokenized_to_id(
    tokenized: List[List[str]],
    word_to_id: "lightautoml.addons.interpretation.utils.WrappedVocabulary",  # noqa F821
    min_k: int,
) -> List[torch.LongTensor]:
    """Mapping from word dataset to tokenized dataset.

    Args:
        tokenized: Dataset of words.
        word_to_id: Dictionary word to token-id.
        min_k: Parameter for minumum sequence length (used for padding).

    Returns:
        Dataset with token ids.

    """
    dataset = []
    for sent in tokenized:
        sent_list = [word_to_id["<START>"]]
        # word_to_id is also callable
        sent_list.extend(map(word_to_id, sent))
        pad_tokens = max(1, min_k - len(sent_list))
        sent_list.extend([word_to_id["<PAD>"]] * pad_tokens)
        dataset.append(torch.Tensor(sent_list).long())

    return dataset


def get_len_dataset(tokenized: List[List[str]], target: np.ndarray) -> LengthDataset:
    """
    Get length dataset.

    Args:
        tokenized: Tokenized strings.
        target: Target values.

    Returns:
        Length Dataset.

    Raises:
        ValueError: Length missmatch of target and tokenized.

    """
    if len(tokenized) != len(target):
        raise ValueError("Missmatch of lengths tokenized ({}) and target ({})".format(len(tokenized), len(target)))
    return LengthDataset(tokenized, target)


def get_len_dataloader(
    dataset: LengthDataset,
    batch_size: int = 1,
    boundaries: Optional[List[int]] = None,
    mode: str = "train",
) -> torch.utils.data.DataLoader:
    """
    Get len dataloader.

    Args:
        dataset: Lenght dataset.
        batch_size: Size of batch.
        boundaries: List of binning boundaries.
        mode: Stage of dataloader.

    Returns:
        Dataloader for L2X.

    """
    if mode == "train":
        sampler = BySequenceLengthSampler(dataset, boundaries, batch_size, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=1, batch_sampler=sampler, collate_fn=pad_max_len)
    elif mode == "valid":
        sampler = BySequenceLengthSampler(dataset, boundaries, batch_size, drop_last=False, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=1, batch_sampler=sampler, collate_fn=pad_max_len)
    elif mode == "test":
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_max_len)

    return dataloader


def create_emb_layer(weights_matrix=None, voc_size=None, embed_dim=None, trainable_embeds=True) -> torch.nn.Embedding:
    """Create initialized embedding layer.

    Args:
        weights_matrix: Weights of embedding layer.
        voc_size: Size of vocabulary.
        embed_dim: Size of embeddings.
        trainable_embeds: To optimize layer when training model.

    Retruns:
        Initialized embedding layer.

    """

    assert (weights_matrix is not None) or (
        voc_size is not None and embed_dim is not None
    ), "Please define anything: weights_matrix or voc_size & embed_dim"

    if weights_matrix is not None:
        voc_size, embed_dim = weights_matrix.size()
    emb_layer = nn.Embedding(voc_size, embed_dim)
    if weights_matrix is not None:
        emb_layer.load_state_dict({"weight": weights_matrix})
    if not trainable_embeds:
        emb_layer.weight.requires_grad = False

    return emb_layer
