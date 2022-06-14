from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np


class GaussianNoise(nn.Module):
    def __init__(self, stddev, device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).to(self.device) * self.stddev)
        return din


class UniformNoise(nn.Module):
    def __init__(self, stddev, device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable((torch.rand(din.size()).to(self.device) - 0.5) * self.stddev)
        return din


class DenseLightBlock(nn.Module):
    def __init__(self, n_in, n_out, drop_rate=0.1, noise_std=0.05, act_fun=nn.ReLU,
                 use_bn=True, use_noise=True, use_dropout=True, use_act=True, device=torch.device("cuda:0"),
                 **kwargs):
        super(DenseLightBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))
        
        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_in))
        if use_dropout:
            self.features.add_module("dropout", nn.Dropout(p=drop_rate))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        self.features.add_module("dense", nn.Linear(n_in, n_out))

        if use_act:
            self.features.add_module("act", act_fun())

    def forward(self, x):
        for name, layer in self.features.named_children():
            x = layer(x)
        return x


class DenseLightModel(nn.Module):
    def __init__(self, n_in, n_out=1, hidden_size=(512, 750,), drop_rate=(0.1, 0.1,),
                 act_fun=nn.ReLU, noise_std=0.05, bias=None, num_init_features=None,
                 use_bn=True, use_noise=True, use_dropout=True, use_act=True,
                 concat_input=True, device=torch.device("cuda:0"), **kwargs):
        super(DenseLightModel, self).__init__()
        assert len(hidden_size) == len(drop_rate), "Wrong number hidden_sizes/drop_rates. Must be equal."

        self.concat_input = concat_input
        num_features = n_in if num_init_features is None else num_init_features
        
        self.features = nn.Sequential(
            OrderedDict(
                []
            )
        )
        if num_init_features is not None:
            self.features.add_module("dense0", nn.Linear(n_in, num_features))
        
        for i, hid_size in enumerate(hidden_size):
            block = DenseLightBlock(
                n_in=num_features,
                n_out=hid_size,
                drop_rate=drop_rate[i] if use_dropout else 0,
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                use_dropout=use_dropout,
                use_act=use_act,
                device=device
            )
            self.features.add_module("denseblock%d" % (i + 1), block)

            if concat_input:
                num_features = n_in + hid_size
            else:
                num_features = hid_size

        num_features = hidden_size[-1]
        self.fc = nn.Linear(num_features, n_out)

        if bias is not None:
            print("init bias!")
            bias = torch.Tensor(bias)
            self.fc.bias.data = bias
            self.fc.weight.data = torch.zeros(n_out, num_features, requires_grad=True)

    def forward(self, x):
        input = x.detach().clone()
        for name, layer in self.features.named_children():
            if name != "denseblock1" and name != "dense0" and self.concat_input:
                x = torch.cat([x, input], 1)
            x = layer(x)

        logits = self.fc(x)
        return logits.view(logits.shape[0], -1)


class MLP(DenseLightModel):
    def __init__(self, **params):
        super(MLP, self).__init__(**{**params, **{'concat_input': False}})

    def forward(self, x):
        return super(MLP, self).forward(x)


class LinearLayer(DenseLightBlock):
    def __init__(self, **params):
        super(LinearLayer, self).__init__(
            **{**params, **{'use_bn': False, 'use_noise': False, 'use_dropout': False, 'use_act': False}})

    def forward(self, x):
        return super(LinearLayer, self).forward(x)


def bn_function_factory(features):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = features(concated_features)
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    def __init__(self, n_in, growth_size=256, bn_factor=2, drop_rate=0.1, act_fun=nn.ReLU,
                 use_bn=True, use_dropout=True, efficient=False, **kwargs):
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


        self.drop_rate = drop_rate
        self.efficient = efficient
        self.use_dropout = use_dropout

    def forward(self, *prev_features):
        bn_function = bn_function_factory(self.features1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        
        new_features = self.features2(bottleneck_output)

        if self.use_dropout:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class Transition(nn.Sequential):
    def __init__(self, n_in, n_out, act_fun, use_bn=True, **kwargs):
        super(Transition, self).__init__()
        if use_bn:
            self.add_module("norm", nn.BatchNorm1d(n_in))
        
        self.add_module("dense", nn.Linear(n_in, n_out))
        self.add_module("act", act_fun())


class DenseBlock(nn.Module):
    def __init__(self, num_layers, n_in, bn_factor, growth_size, drop_rate=0.1, act_fun=nn.ReLU,
                 use_bn=True, use_dropout=True, efficient=False, **kwargs):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                n_in + i * growth_size,
                growth_size=growth_size,
                bn_factor=bn_factor,
                drop_rate=drop_rate,
                act_fun=act_fun,
                efficient=efficient,
                use_bn=use_bn,
                use_dropout=use_dropout
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseModel(nn.Module):
    def __init__(self, n_in, n_out=1, block_config=(2, 2), drop_rate=(0.1, 0.1), num_init_features=None,
                 compression=0.5, growth_size=256, bn_factor=2, bias=None, act_fun=nn.ReLU,
                 use_bn=True, use_dropout=True, efficient=False, **kwargs):

        super(DenseModel, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"
        
        num_features = n_in if num_init_features is None else num_init_features
        self.features = nn.Sequential(
            OrderedDict(
                []
            )
        )
        if num_init_features is not None:
            self.features.add_module("dense0", nn.Linear(n_in, num_features))

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                n_in=num_features,
                bn_factor=bn_factor,
                growth_size=growth_size,
                drop_rate=drop_rate[i] if use_dropout else 0,
                act_fun=act_fun,
                efficient=efficient,
                use_bn=use_bn,
                use_dropout=use_dropout
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_size
            if i != len(block_config) - 1:
                trans = Transition(n_in=num_features,
                                   n_out=max(10, int(num_features * compression)),
                                   act_fun=act_fun,
                                   use_bn=use_bn)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = max(10, int(num_features * compression))

        if use_bn:
            self.features.add_module("norm_final", nn.BatchNorm1d(num_features))

        self.fc = nn.Linear(num_features, n_out)
        
        if bias is not None:
            print("init bias!")
            bias = torch.Tensor(bias)
            self.fc.bias.data = bias
            self.fc.weight.data = torch.zeros(n_out, num_features, requires_grad=True)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, n_in, hid_factor, n_out, drop_rate=(0.1, 0.1), noise_std=0.05, act_fun=nn.ReLU,
                 use_bn=True, use_noise=True, use_dropout=True, device=torch.device("cuda:0"), **kwargs):
        super(ResNetBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))
        
        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_in))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        self.features.add_module("dense1", nn.Linear(n_in, int(hid_factor * n_in)))
        self.features.add_module("act1", act_fun())
                                 
        if use_dropout:
            self.features.add_module("drop1", nn.Dropout(p=drop_rate[0]))
        
        self.features.add_module("dense2", nn.Linear(int(hid_factor * n_in), n_out))
                                 
        if use_dropout:
            self.features.add_module("drop2", nn.Dropout(p=drop_rate[1]))


    def forward(self, x):
        x = self.features(x)
        return x


class ResNetModel(nn.Module):
    """The ResNet model from 
        https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py#L410
    """
    def __init__(self, n_in, n_out=1, hid_factor=(2, 2), drop_rate=((0.1, 0.1), (0.1, 0.1)),
                 bias=None, noise_std=0.05, act_fun=nn.ReLU, num_init_features=None,
                 use_bn=True, use_noise=True, use_dropout=True, device=torch.device("cuda:0"),
                 **kwargs):
        super(ResNetModel, self).__init__()
        num_features = n_in if num_init_features is None else num_init_features
        self.dense0 = nn.Linear(n_in, num_features) if num_init_features is not None else nn.Identity()
        self.features1 = nn.Sequential(OrderedDict([]))
        
        for i, hd_factor in enumerate(hid_factor):
            block = ResNetBlock(
                n_in=num_features,
                hid_factor=hd_factor,
                n_out=num_features,
                drop_rate=drop_rate[i] if use_dropout else 0,
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                use_dropout=use_dropout,
                device=device
            )
            self.features1.add_module("resnetblock%d" % (i + 1), block)
        
        self.features2 = nn.Sequential(OrderedDict([]))
        if use_bn:
            self.features2.add_module("norm", nn.BatchNorm1d(num_features))
        
        self.features2.add_module("act", act_fun())
        self.fc = nn.Linear(num_features, n_out)

        if bias is not None:
            print("init bias!")
            bias = torch.Tensor(bias)
            self.fc.bias.data = bias
            self.fc.weight.data = torch.zeros(n_out, num_features, requires_grad=True)

    def forward(self, x):
        x = self.dense0(x)
        identity = x
        for name, layer in self.features1.named_children():
            if name != "resnetblock1":
                x += identity
                identity = x
            x = layer(x)
        
        x = self.features2(x)
        logits = self.fc(x)
        return logits.view(logits.shape[0], -1)


class SNN(nn.Module):
    def __init__(self, n_in, n_out, hidden_size=512, num_layers=3, drop_rate=0.1,
                 num_init_features=None, use_dropout=True, **kwargs):
        super().__init__()
        num_features = n_in if num_init_features is None else num_init_features
        
        layers = OrderedDict([])
        self.dense0 = nn.Linear(n_in, num_features) if num_init_features is not None else nn.Identity()
        
        i = 0
        while i != num_layers:
            if i == 0:
                layers[f"dense{i}"] = nn.Linear(num_features, hidden_size, bias=False)
            else:
                layers[f"dense{i}"] = nn.Linear(hidden_size, hidden_size, bias=False)
            layers[f"selu_{i}"] = nn.SELU()

            if use_dropout:
                layers[f"dropout_{i}"] = nn.AlphaDropout(p=drop_rate)
            i += 1
        
        layers[f"dense_{i}"] = nn.Linear(hidden_size, n_out, bias=True)
        self.network = nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x):
        x = self.dense0(x)
        x = self.network(x)
        x = x.view(x.shape[0], -1)
        return x

    def reset_parameters(self):
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / np.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

