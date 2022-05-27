"""Neural Net modules for differen data types."""


from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


try:
    from transformers import AutoModel
except:
    import warnings

    warnings.warn("'transformers' - package isn't installed")

from ..tasks.base import Task
from .dl_transformers import pooling_by_name


class UniversalDataset:
    """Dataset class for mixed data."""

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        tokenizer: Optional = None,
        max_length: int = 256,
        stage: str = "test",
    ):
        """Class for preparing input for DL model with mixed data.

        Args:
            data: Dict with data.
            y: Array of target variable.
            w: Optional array of observation weight.
            tokenizer: Transformers tokenizer.
            max_length: Max sentence length.
            stage: Name of current training / inference stage.

        """
        self.data = data
        self.y = y
        self.w = w
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        res = {"label": self.y[index]}
        res.update({key: value[index] for key, value in self.data.items() if key != "text"})
        if (self.tokenizer is not None) and ("text" in self.data):
            sent = self.data["text"][index, 0]  # only one column
            _split = sent.split("[SEP]")
            sent = _split if len(_split) == 2 else (sent,)
            data = self.tokenizer.encode_plus(
                *sent, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True
            )

            res.update({i: np.array(data[i]) for i in data.keys()})
        if self.w is not None:
            res["weight"] = self.w[index]

        return res


class Clump(nn.Module):
    """Clipping input tensor."""

    def __init__(self, min_v: int = -50, max_v: int = 50):
        """Class for preparing input for DL model with mixed data.

        Args:
            min_v: Min value.
            max_v: Max value.

        """
        super(Clump, self).__init__()

        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.min_v, self.max_v)
        return x


class TextBert(nn.Module):
    """Text data model."""

    _poolers = {"cls", "max", "mean", "sum", "none"}

    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "cls"):
        """Class for working with text data based on HuggingFace transformers.

        Args:
            model_name: Transformers model name.
            pooling: Pooling type.

        Note:
            There are different pooling types:

                - cls: Use CLS token for sentence embedding
                  from last hidden state.
                - max: Maximum on seq_len dimension for non masked
                  inputs from last hidden state.
                - mean: Mean on seq_len dimension for non masked
                  inputs from last hidden state.
                - sum: Sum on seq_len dimension for non masked
                  inputs from last hidden state.
                - none: Without pooling for seq2seq models.

        """
        super(TextBert, self).__init__()
        if pooling not in self._poolers:
            raise ValueError("pooling - {} - not in the list of available types {}".format(pooling, self._poolers))

        self.transformer = AutoModel.from_pretrained(model_name)
        self.n_out = self.transformer.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU(inplace=True)
        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        # last hidden layer
        encoded_layers, _ = self.transformer(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            token_type_ids=inp.get("token_type_ids"),
            return_dict=False,
        )

        # pool the outputs into a vector
        encoded_layers = self.pooling(encoded_layers, inp["attention_mask"].unsqueeze(-1).bool())
        mean_last_hidden_state = self.activation(encoded_layers)
        mean_last_hidden_state = self.dropout(mean_last_hidden_state)
        return mean_last_hidden_state


class CatEmbedder(nn.Module):
    """Category data model."""

    def __init__(
        self,
        cat_dims: Sequence[int],
        emb_dropout: bool = 0.1,
        emb_ratio: int = 3,
        max_emb_size: int = 50,
    ):
        """Class for working with category data using embedding layer.

        Args:
            cat_dims: Sequence with number of unique categories
              for category features.
            emb_dropout: Dropout probability.
            emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
            max_emb_size: Max embedding size.

        """
        super(CatEmbedder, self).__init__()
        emb_dims = [(int(x), int(min(max_emb_size, max(1, (x + 1) // emb_ratio)))) for x in cat_dims]
        self.no_of_embs = sum([y for x, y in emb_dims])
        assert self.no_of_embs != 0, "The input is empty."
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.no_of_embs

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = torch.cat(
            [emb_layer(inp["cat"][:, i]) for i, emb_layer in enumerate(self.emb_layers)],
            dim=1,
        )
        output = self.emb_dropout_layer(output)
        return output


class ContEmbedder(nn.Module):
    """Numeric data model."""

    def __init__(self, num_dims: int, input_bn: bool = True):
        """Class for working with numeric data.

        Args:
            num_dims: Sequence with number of numeric features.
            input_bn: Use 1d batch norm for input data.

        """
        super(ContEmbedder, self).__init__()
        self.n_out = num_dims
        self.bn = None
        if input_bn:
            self.bn = nn.BatchNorm1d(num_dims)
        assert num_dims != 0, "The input is empty."

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = inp["cont"]
        if self.bn is not None:
            output = self.bn(output)
        return output


class TorchUniversalModel(nn.Module):
    """Mixed data model."""

    def __init__(
        self,
        loss: Callable,
        task: Task,
        n_out: int = 1,
        cont_embedder: Optional = None,
        cont_params: Optional[Dict] = None,
        cat_embedder: Optional = None,
        cat_params: Optional[Dict] = None,
        text_embedder: Optional = None,
        text_params: Optional[Dict] = None,
        bias: Optional[Sequence] = None,
    ):
        """Class for preparing input for DL model with mixed data.

        Args:
            loss: Callable torch loss with order of arguments (y_true, y_pred).
            task: Task object.
            n_out: Number of output dimensions.
            cont_embedder: Torch module for numeric data.
            cont_params: Dict with numeric model params.
            cat_embedder: Torch module for category data.
            cat_params: Dict with category model params.
            text_embedder: Torch module for text data.
            text_params: Dict with text model params.
            bias: Array with last hidden linear layer bias.

        """
        super(TorchUniversalModel, self).__init__()
        self.n_out = n_out
        self.loss = loss
        self.task = task

        self.cont_embedder = None
        self.cat_embedder = None
        self.text_embedder = None

        n_in = 0
        if cont_embedder is not None:
            self.cont_embedder = cont_embedder(**cont_params)
            n_in += self.cont_embedder.get_out_shape()
        if cat_embedder is not None:
            self.cat_embedder = cat_embedder(**cat_params)
            n_in += self.cat_embedder.get_out_shape()
        if text_embedder is not None:
            self.text_embedder = text_embedder(**text_params)
            n_in += self.text_embedder.get_out_shape()

        self.bn = nn.BatchNorm1d(n_in)
        self.fc = torch.nn.Linear(n_in, self.n_out)

        if bias is not None:
            bias = torch.Tensor(bias)
            self.fc.bias.data = nn.Parameter(bias)
            self.fc.weight.data = nn.Parameter(torch.zeros(self.n_out, n_in))

        if (self.task.name == "binary") or (self.task.name == "multilabel"):
            self.fc = nn.Sequential(self.fc, Clump(), nn.Sigmoid())
        elif self.task.name == "multiclass":
            self.fc = nn.Sequential(self.fc, Clump(), nn.Softmax(dim=1))

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.predict(inp)
        loss = self.loss(inp["label"].view(inp["label"].shape[0], -1), x, inp.get("weight", None))
        return loss

    def predict(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = []
        if self.cont_embedder is not None:
            outputs.append(self.cont_embedder(inp))

        if self.cat_embedder is not None:
            outputs.append(self.cat_embedder(inp))

        if self.text_embedder is not None:
            outputs.append(self.text_embedder(inp))

        if len(outputs) > 1:
            output = torch.cat(outputs, dim=1)
        else:
            output = outputs[0]
        logits = self.fc(output)
        return logits.view(logits.shape[0], -1)
