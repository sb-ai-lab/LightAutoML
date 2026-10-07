import numpy as np

from lightautoml.text import embed_dataset
from lightautoml.text.nn_model import UniversalDataset


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, *texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


def test_universal_dataset_uses_tokenizer_call():
    tokenizer = FakeTokenizer()
    dataset = UniversalDataset({"text": np.array([["first[SEP]second"]])}, np.array([1]), tokenizer=tokenizer)

    sample = dataset[0]

    assert tokenizer.calls[0][0] == ("first", "second")
    np.testing.assert_array_equal(sample["input_ids"], [1, 2])


def test_bert_dataset_uses_tokenizer_call(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(embed_dataset.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    dataset = embed_dataset.BertDataset(["text"], max_length=8, model_name="test")

    sample = dataset[0]

    assert tokenizer.calls[0][0] == ("text",)
    np.testing.assert_array_equal(sample["attention_mask"], [1, 1])
