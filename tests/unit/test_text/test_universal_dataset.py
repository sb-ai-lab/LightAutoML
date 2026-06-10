import numpy as np
import torch

from lightautoml.text.nn_model import UniversalDataset
from lightautoml.text.utils import collate_dict


def test_universal_dataset_batched_getitems_matches_rowwise_batch():
    data = {
        "cat": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64),
        "cont": np.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]], dtype=np.float32),
    }
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    w = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32)
    indices = [0, 2, 3]

    dataset = UniversalDataset(data=data, y=y, w=w, tokenizer=None)

    row_batch = [dataset[index] for index in indices]
    fast_batch = dataset.__getitems__(indices)
    row_collated = collate_dict(row_batch)
    fast_collated = collate_dict(fast_batch)

    assert row_collated.keys() == fast_collated.keys()
    for key in row_collated:
        assert row_collated[key].shape == fast_collated[key].shape
        assert torch.equal(row_collated[key], fast_collated[key])

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
    class DummyTokenizer:
        def encode_plus(self, *args, **kwargs):
            return {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
            }

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
