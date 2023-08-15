"""Torch models."""

from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from lightautoml.ml_algo.torch_based.node_nn_model import DenseODSTBlock, MeanPooling


class GaussianNoise(nn.Module):
    """Adds gaussian noise.

    Args:
        stddev: Std of noise.
        device: Device to compute on.

    """

    def __init__(self, stddev: float, device: torch.device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        if self.training:
            return x + torch.randn(x.size(), device=self.device) * self.stddev
        return x


class UniformNoise(nn.Module):
    """Add uniform noise.

    Args:
            stddev: Std of noise.
            device: Device to compute on.

    """

    def __init__(self, stddev: float, device: torch.Tensor):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        if self.training:
            return x + (torch.rand(x.size(), device=self.device) - 0.5) * self.stddev
        return x


class DenseLightBlock(nn.Module):
    """Realisation of `'denselight'` model block.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            drop_rate: Dropout rate.
            noise_std: Std of noise.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        drop_rate: float = 0.1,
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(DenseLightBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_in))
        if drop_rate:
            self.features.add_module("dropout", nn.Dropout(p=drop_rate))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        self.features.add_module("dense", nn.Linear(n_in, n_out))
        self.features.add_module("act", act_fun())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        for name, layer in self.features.named_children():
            x = layer(x)
        return x


class DenseLightModel(nn.Module):
    """Realisation of `'denselight'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            drop_rate: Dropout rate for each layer separately or altogether.
            act_fun: Activation function.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            concat_input: Concatenate input to all hidden layers.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hidden_size: List[int] = [
            512,
            750,
        ],
        drop_rate: Union[float, List[float]] = 0.1,
        act_fun: nn.Module = nn.ReLU,
        noise_std: float = 0.05,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        use_noise: bool = False,
        concat_input: bool = True,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(DenseLightModel, self).__init__()

        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * len(hidden_size)

        assert len(hidden_size) == len(drop_rate), "Wrong number hidden_sizes/drop_rates. Must be equal."

        self.concat_input = concat_input
        num_features = n_in if num_init_features is None else num_init_features

        self.features = nn.Sequential(OrderedDict([]))
        if num_init_features is not None:
            self.features.add_module("dense0", nn.Linear(n_in, num_features))

        for i, hid_size in enumerate(hidden_size):
            block = DenseLightBlock(
                n_in=num_features,
                n_out=hid_size,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                device=device,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)

            if concat_input:
                num_features = n_in + hid_size
            else:
                num_features = hid_size

        num_features = hidden_size[-1]
        self.fc = nn.Linear(num_features, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        input = x.detach().clone()
        for name, layer in self.features.named_children():
            if name != "denseblock1" and name != "dense0" and self.concat_input:
                x = torch.cat([x, input], 1)
            x = layer(x)
        x = self.fc(x)
        return x


class MLP(DenseLightModel):
    """Realisation of `'mlp'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            drop_rate: Dropout rate for each layer separately or altogether.
            act_fun: Activation function.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(*args, **{**kwargs, **{"concat_input": False}})


class _LinearLayer(DenseLightBlock):
    """Realisation of `'_linear_layer'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            device: Device to compute on.

    """

    def __init__(self, *args, **kwargs):
        super(_LinearLayer, self).__init__(
            *args,
            **{
                **kwargs,
                **{
                    "use_bn": True,
                    "use_noise": False,
                    "drop_rate": 0.0,
                    "act_fun": nn.Identity,
                },
            },
        )


class LinearLayer(DenseLightBlock):
    """Realisation of `'linear_layer'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            device: Device to compute on.

    """

    def __init__(self, *args, **kwargs):
        super(LinearLayer, self).__init__(
            *args,
            **{
                **kwargs,
                **{
                    "use_bn": False,
                    "use_noise": False,
                    "drop_rate": 0.0,
                    "act_fun": nn.Identity,
                },
            },
        )


class DenseLayer(nn.Module):
    """Realisation of `'dense'` model layer.

    Args:
            n_in: Input dim.
            growth_size: Output dim.
            bn_factor: Dim of intermediate fc is increased times `bn_factor` in DenseModel layer.
            drop_rate: Dropout rate.
            act_fun: Activation function.
            use_bn: Use BatchNorm.

    """

    def __init__(
        self,
        n_in: int,
        growth_size: int = 256,
        bn_factor: float = 2,
        drop_rate: float = 0.1,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        **kwargs,
    ):
        super(DenseLayer, self).__init__()

        self.features1 = nn.Sequential(OrderedDict([]))
        self.features2 = nn.Sequential(OrderedDict([]))

        if use_bn:
            self.features1.add_module("norm1", nn.BatchNorm1d(n_in))

        self.features1.add_module("dense1", nn.Linear(n_in, int(bn_factor * n_in)))
        self.features1.add_module("act1", act_fun())

        if use_bn:
            self.features2.add_module("norm2", nn.BatchNorm1d(int(bn_factor * n_in)))

        self.features2.add_module("dense2", nn.Linear(int(bn_factor * n_in), growth_size))
        self.features2.add_module("act2", act_fun())

        if drop_rate:
            self.features2.add_module("dropout", nn.Dropout(drop_rate))

    def forward(self, prev_features: List[torch.Tensor]):
        """Forward-pass."""
        x = self.features1(torch.cat(prev_features, 1))
        x = self.features2(x)
        return x


class Transition(nn.Sequential):
    """Compress input to lower dim.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            growth_size: Output dim of every layer.
            act_fun: Activation function.
            use_bn: Use BatchNorm.

    """

    def __init__(self, n_in: int, n_out: int, act_fun: nn.Module, use_bn: bool = True):
        super(Transition, self).__init__()
        if use_bn:
            self.add_module("norm", nn.BatchNorm1d(n_in))

        self.add_module("dense", nn.Linear(n_in, n_out))
        self.add_module("act", act_fun())


class DenseBlock(nn.Module):
    """Realisation of `'dense'` model block.

    Args:
            n_in: Input dim.
            num_layers: Number of layers.
            bn_factor: Dim of intermediate fc is increased times `bn_factor` in DenseModel layer.
            growth_size: Output dim of every layer.
            drop_rate: Dropout rate.
            act_fun: Activation function.
            use_bn: Use BatchNorm.

    """

    def __init__(
        self,
        num_layers: int,
        n_in: int,
        bn_factor: float,
        growth_size: int,
        drop_rate: float = 0.1,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        **kwargs,
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                n_in + i * growth_size,
                growth_size=growth_size,
                bn_factor=bn_factor,
                drop_rate=drop_rate,
                act_fun=act_fun,
                use_bn=use_bn,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: List[torch.Tensor]):
        """Forward-pass with layer output concatenation in the end."""
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseModel(nn.Module):
    """Realisation of `'dense'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            block_config: List of number of layers within each block
            drop_rate: Dropout rate for each layer separately or altogether.
            num_init_features: If not none add fc layer before model with certain dim.
            compression: portion of neuron to drop after block.
            growth_size: Output dim of every layer.
            bn_factor: Dim of intermediate fc is increased times `bn_factor` in DenseModel layer.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        block_config: List[int] = [2, 2],
        drop_rate: Union[float, List[float]] = 0.1,
        num_init_features: Optional[int] = None,
        compression: float = 0.5,
        growth_size: int = 256,
        bn_factor: float = 2,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        **kwargs,
    ):
        super(DenseModel, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"

        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * len(block_config)

        assert len(block_config) == len(drop_rate), "Wrong number hidden_sizes/drop_rates. Must be equal."

        num_features = n_in if num_init_features is None else num_init_features
        self.features = nn.Sequential(OrderedDict([]))
        if num_init_features is not None:
            self.features.add_module("dense0", nn.Linear(n_in, num_features))

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                n_in=num_features,
                bn_factor=bn_factor,
                growth_size=growth_size,
                drop_rate=drop_rate[i],
                act_fun=act_fun,
                use_bn=use_bn,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_size
            if i != len(block_config) - 1:
                trans = Transition(
                    n_in=num_features,
                    n_out=max(10, int(num_features * compression)),
                    act_fun=act_fun,
                    use_bn=use_bn,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = max(10, int(num_features * compression))

        if use_bn:
            self.features.add_module("norm_final", nn.BatchNorm1d(num_features))

        self.fc = nn.Linear(num_features, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNetBlock(nn.Module):
    """Realisation of `'resnet'` model block.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hid_factor: Dim of intermediate fc is increased times this factor in ResnetModel layer.
            drop_rate: Dropout rates.
            noise_std: Std of noise.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        hid_factor: float,
        n_out: int,
        drop_rate: List[float] = [0.1, 0.1],
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(ResNetBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_in))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        self.features.add_module("dense1", nn.Linear(n_in, int(hid_factor * n_in)))
        self.features.add_module("act1", act_fun())

        if drop_rate[0]:
            self.features.add_module("drop1", nn.Dropout(p=drop_rate[0]))

        self.features.add_module("dense2", nn.Linear(int(hid_factor * n_in), n_out))

        if drop_rate[1]:
            self.features.add_module("drop2", nn.Dropout(p=drop_rate[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.features(x)
        return x


class ResNetModel(nn.Module):
    """The ResNet model from https://github.com/Yura52/rtdl.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hid_factor: Dim of intermediate fc is increased times this factor in ResnetModel layer.
            drop_rate: Dropout rate for each layer separately or altogether.
            noise_std: Std of noise.
            act_fun: Activation function.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hid_factor: List[float] = [2, 2],
        drop_rate: Union[float, List[float], List[List[float]]] = 0.1,
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(ResNetModel, self).__init__()
        if isinstance(drop_rate, float):
            drop_rate = [[drop_rate, drop_rate]] * len(hid_factor)
        elif isinstance(drop_rate, list) and len(drop_rate) == 2:
            drop_rate = [drop_rate] * len(hid_factor)
        else:
            assert (
                len(drop_rate) == len(hid_factor) and len(drop_rate[0]) == 2
            ), "Wrong number hidden_sizes/drop_rates. Must be equal."

        num_features = n_in if num_init_features is None else num_init_features
        self.dense0 = nn.Linear(n_in, num_features) if num_init_features is not None else nn.Identity()
        self.features1 = nn.Sequential(OrderedDict([]))

        for i, hd_factor in enumerate(hid_factor):
            block = ResNetBlock(
                n_in=num_features,
                hid_factor=hd_factor,
                n_out=num_features,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                device=device,
            )
            self.features1.add_module("resnetblock%d" % (i + 1), block)

        self.features2 = nn.Sequential(OrderedDict([]))
        if use_bn:
            self.features2.add_module("norm", nn.BatchNorm1d(num_features))

        self.features2.add_module("act", act_fun())
        self.fc = nn.Linear(num_features, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.dense0(x)
        identity = x
        for name, layer in self.features1.named_children():
            if name != "resnetblock1":
                x += identity
                identity = x
            x = layer(x)

        x = self.features2(x)
        x = self.fc(x)
        return x.view(x.shape[0], -1)


class SNN(nn.Module):
    """Realisation of `'snn'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            drop_rate: Dropout rate for each layer separately or altogether.
            num_init_features: If not none add fc layer before model with certain dim.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_size: List[int] = [512, 512, 512],
        num_init_features: Optional[int] = None,
        drop_rate: Union[float, List[float]] = 0.1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * len(hidden_size)

        num_features = n_in if num_init_features is None else num_init_features
        layers = OrderedDict([])
        if num_init_features is not None:
            layers["dense-1"] = nn.Linear(n_in, num_features, bias=False)

        for i in range(len(hidden_size) - 1):
            layers[f"dense{i}"] = nn.Linear(num_features, hidden_size[i], bias=False)
            layers[f"selu_{i}"] = nn.SELU()

            if drop_rate[i]:
                layers[f"dropout_{i}"] = nn.AlphaDropout(p=drop_rate[i])
            num_features = hidden_size[i]

        layers[f"dense_{i}"] = nn.Linear(hidden_size[i], n_out, bias=True)
        self.network = nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.network(x)
        x = x.view(x.shape[0], -1)
        return x

    def reset_parameters(self):
        """Init weights."""
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / np.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)


"""Different Pooling strategies for sequence data."""


class SequenceAbstractPooler(nn.Module):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Forward-call."""
        return self.forward(*args, **kwargs)


class SequenceClsPooler(SequenceAbstractPooler):
    """CLS token pooling."""

    def __init__(self):
        super(SequenceClsPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        return x[..., 0, :]


class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, -float("inf"))
        values, _ = torch.max(x, dim=-2)
        return values


class SequenceSumPooler(SequenceAbstractPooler):
    """Sum value pooling."""

    def __init__(self):
        super(SequenceSumPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, 0)
        values = torch.sum(x, dim=-2)
        return values


class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active.data
        return values


class SequenceIndentityPooler(SequenceAbstractPooler):
    """Identity pooling."""

    def __init__(self):
        super(SequenceIndentityPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        return x


class NODE(nn.Module):
    """The NODE model from https://github.com/Qwicen.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            layer_dim: num trees in one layer.
            num_layers: number of forests.
            tree_dim: number of response channels in the response of individual tree.
            use_original_head use averaging as a head or put linear layer instead.
            depth: number of splits in every tree.
            drop_rate: Dropout rate for each layer altogether.
            act_fun: Activation function.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        layer_dim: int = 2048,
        num_layers: int = 1,
        tree_dim: int = 1,
        use_original_head: bool = False,
        depth: int = 6,
        drop_rate: float = 0.0,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        **kwargs,
    ):
        super(NODE, self).__init__()
        num_features = n_in if num_init_features is None else num_init_features
        self.dense0 = nn.Linear(n_in, num_features) if num_init_features is not None else nn.Identity()
        self.features1 = nn.Sequential(OrderedDict([]))
        block = DenseODSTBlock(
            input_dim=num_features,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=tree_dim if not use_original_head else n_out,
            depth=depth,
            input_dropout=drop_rate,
            flatten_output=not use_original_head,
        )
        self.features1.add_module("ODSTForestblock%d", block)
        self.features2 = nn.Sequential(OrderedDict([]))
        if use_original_head:
            last_layer = MeanPooling(n_out, dim=-2)
            self.features2.add_module("head", last_layer)
        else:
            if use_bn:
                self.features2.add_module("norm", nn.BatchNorm1d(layer_dim * num_layers * tree_dim))
            self.features2.add_module("act", act_fun())
            fc = nn.Linear(layer_dim * num_layers * tree_dim, n_out)
            self.features2.add_module("fc", fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.dense0(x)
        x = self.features1(x)
        x = self.features2(x)
        return x.view(x.shape[0], -1)
