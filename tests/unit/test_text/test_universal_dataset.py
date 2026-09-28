import numpy as np
import torch
from torch.utils.data import DataLoader

from lightautoml.text.nn_model import UniversalDataset
from lightautoml.text.utils import collate_dict


class DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
        }


def _tabular_dataset():
    data = {
        "cat": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64),
        "cont": np.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]], dtype=np.float32),
    }
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    w = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32)
    return UniversalDataset(data=data, y=y, w=w, tokenizer=None)


def _assert_batch_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].shape == right[key].shape
        assert torch.equal(left[key], right[key])


def test_universal_dataset_batched_getitems_matches_rowwise_batch():
    dataset = _tabular_dataset()
    indices = [0, 2, 3]
    row_collated = collate_dict([dataset[index] for index in indices])
    fast_collated = collate_dict(dataset.__getitems__(indices))

    _assert_batch_equal(row_collated, fast_collated)
    assert fast_collated["cat"].dtype == torch.int64
    assert fast_collated["cont"].dtype == torch.float32
    assert fast_collated["label"].dtype == torch.float32
    assert fast_collated["weight"].dtype == torch.float32


def test_collate_dict_supports_rowwise_and_batched_dict_inputs():
    row_batch = [
        {
            "cat": np.array([1, 2], dtype=np.int64),
            "cont": np.array([0.1, 0.2], dtype=np.float32),
            "label": np.array(0.0, dtype=np.float32),
            "weight": np.array(1.0, dtype=np.float32),
        },
        {
            "cat": np.array([3, 4], dtype=np.int64),
            "cont": np.array([1.1, 1.2], dtype=np.float32),
            "label": np.array(1.0, dtype=np.float32),
            "weight": np.array(0.5, dtype=np.float32),
        },
    ]
    batched_dict = {
        "cat": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "cont": np.array([[0.1, 0.2], [1.1, 1.2]], dtype=np.float32),
        "label": np.array([0.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 0.5], dtype=np.float32),
    }

    row_collated = collate_dict(row_batch)
    batched_collated = collate_dict(batched_dict)

    assert row_collated.keys() == batched_collated.keys()
    for key in row_collated:
        assert row_collated[key].shape == batched_collated[key].shape
        assert torch.equal(row_collated[key], batched_collated[key])

    assert batched_collated["cat"].dtype == torch.int64
    assert batched_collated["cont"].dtype == torch.float32
    assert batched_collated["label"].dtype == torch.float32
    assert batched_collated["weight"].dtype == torch.float32


def test_universal_dataset_getitems_falls_back_to_rowwise_with_tokenizer():
    dataset = UniversalDataset(
        data={"text": np.array([["hello"], ["world"]])},
        y=np.array([0.0, 1.0], dtype=np.float32),
        tokenizer=DummyTokenizer(),
    )

    batch = dataset.__getitems__([0, 1])
    row_batch = [dataset[0], dataset[1]]

    assert isinstance(batch, list)
    assert len(batch) == 2
    for batched_item, row_item in zip(batch, row_batch):
        assert batched_item.keys() == row_item.keys()
        for key in batched_item:
            assert np.array_equal(batched_item[key], row_item[key])


def test_dataloader_tabular_matches_rowwise_collate():
    dataset = _tabular_dataset()
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_dict)))
    expected = collate_dict([dataset[i] for i in range(3)])
    _assert_batch_equal(batch, expected)


def test_dataloader_tokenizer_returns_collated_batch():
    dataset = UniversalDataset(
        data={"text": np.array([["hello"], ["world"], ["sep"]])},
        y=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        tokenizer=DummyTokenizer(),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_dict)))
    expected = collate_dict([dataset[0], dataset[1]])
    _assert_batch_equal(batch, expected)
    assert batch["input_ids"].dtype == torch.int64


def test_getitems_accepts_numpy_indices():
    dataset = _tabular_dataset()
    indices = np.array([0, 2, 3])
    _assert_batch_equal(
        collate_dict(dataset.__getitems__(indices)),
        collate_dict([dataset[int(i)] for i in indices]),
    )


def test_getitems_matches_rowwise_for_2d_tabm_indices():
    dataset = _tabular_dataset()
    indices = np.array([[0, 2, 3], [1, 1, 0]])  # (batch, k)
    fast = collate_dict(dataset.__getitems__(indices))
    _assert_batch_equal(fast, collate_dict([dataset[row] for row in indices]))
    assert fast["cat"].shape == (2, 3, 3)
