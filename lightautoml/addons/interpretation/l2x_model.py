from typing import Optional

import torch
import torch.nn.functional as F

from torch import nn
from torch.distributions.utils import clamp_probs

from .data_process import create_emb_layer


class TIModel(nn.Module):
    def __init__(
        self,
        voc_size: int,
        embed_dim: int = 50,
        conv_filters: int = 100,
        conv_ksize: int = 3,
        drop_rate: float = 0.2,
        hidden_dim: int = 100,
        weights_matrix: Optional[torch.FloatTensor] = None,
        trainable_embeds: bool = False,
    ):
        super(TIModel, self).__init__()

        self.lookup = create_emb_layer(weights_matrix, voc_size, embed_dim, trainable_embeds)

        embed_dim = self.lookup.embedding_dim
        self.drop1 = nn.Dropout(p=drop_rate)
        self.conv1 = nn.Conv1d(embed_dim, conv_filters, conv_ksize, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool1d(output_size=1)
        self.global_info = nn.Linear(conv_filters, hidden_dim)
        self.global_act = nn.ReLU()
        self.conv2 = nn.Conv1d(conv_filters, hidden_dim, conv_ksize, padding=1)
        self.act2 = nn.ReLU()
        self.local_info = nn.Conv1d(hidden_dim, hidden_dim, conv_ksize, padding=1)
        self.local_act = nn.ReLU()
        self.drop3 = nn.Dropout(p=drop_rate)
        self.conv3 = nn.Conv1d(2 * hidden_dim, conv_filters, 1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv1d(conv_filters, 1, 1)

    def get_embedding(self, input):
        return self.lookup(input)

    def get_embedding_layer(self):
        return self.lookup

    def freeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(False)

    def unfreeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(True)

    def predict(self, x):
        # input: batch_size x len_seq x embed_dim
        # batch_size x embed_dim x len_seq
        x = x.transpose(1, 2)
        # batch_size x conv_filters x len_seq
        x = self.act1(self.conv1(self.drop1(x)))
        # batch_size x conv_filters x len_seq -> batch_size x conv_filters ->
        # -> batch_size x hidden_dim
        global_info = self.global_act(self.global_info(self.pool1(x).squeeze(2)))
        # batch_size x hidden_dim x len_seq ->
        # -> batch_size x hidden_dim x len_seq
        local_info = self.local_act(self.local_info(self.act2(self.conv2(x))))
        # batch_size x hidden_dim x 1 ->
        # -> batch_size x hidden_dim x len_seq
        global_info = global_info.unsqueeze(-1).expand_as(local_info)
        # batch_size x 2 * hidden_dim x len_seq
        z = torch.cat([global_info, local_info], dim=1)
        # batch_size x conv_filters x len_seq
        z = self.act3(self.conv3(self.drop3(z)))
        # batch_size x 1 x len_seq
        logits = self.conv4(z)

        return logits

    def forward(self, x):
        embed = self.get_embedding(x)
        logits = self.predict(embed)

        return logits


class GumbelTopKSampler(nn.Module):
    def __init__(self, T, k):
        super(GumbelTopKSampler, self).__init__()
        # Use as parameter to serialize,
        # Serialization of pytorch, when loading an already initialized model
        # does not load class arguments, only arguments that are model parameters.
        # So code (pseudo-code) like this, doesn't work right:
        # >>> l2x = L2XModel(...)
        # >>> for epoch in range(n_epochs):
        # >>>     train_step(...)
        # >>>     val_loss = valid_step(...)
        # >>>     if best_loss > val_loss:
        # >>>         torch.save(l2x.state_dict(), self._checkpoint_path)
        # >>> l2x.load_state_dict(torch.load(self._checkpoint_path))
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.int32), requires_grad=False)

    def sample_continous(self, logits):
        l_shape = (logits.shape[0], self.k, logits.shape[2])
        u = clamp_probs(torch.rand(l_shape, device=logits.device))
        gumbel = -torch.log(-torch.log(u))
        noisy_logits = (gumbel + logits) / self.T
        samples = F.softmax(noisy_logits, dim=-1)
        samples = torch.max(samples, dim=1)[0]

        return samples

    def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()

        return samples

    def forward(self, logits):
        csamples = None
        if self.training:
            csamples = self.sample_continous(logits)

        dsamples = self.sample_discrete(logits)

        return dsamples, csamples


class SoftSubSampler(nn.Module):
    def __init__(self, T, k):
        super(SoftSubSampler, self).__init__()
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.int32), requires_grad=False)

    def inject_noise(self, logits):
        u = clamp_probs(torch.rand_like(logits))
        z = -torch.log(-torch.log(u))
        noisy_logits = logits + z
        return noisy_logits

    def continuous_topk(self, w, separate=False):
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype=torch.float32)
        for _ in range(self.k):
            khot_mask = clamp_probs(1.0 - onehot_approx)
            w += torch.log(khot_mask)
            onehot_approx = F.softmax(w / self.T, dim=-1)
            khot_list.append(onehot_approx)
        if separate:
            return khot_list
        else:
            return torch.stack(khot_list, dim=-1).sum(-1).squeeze(1)

    def sample_continous(self, logits):
        return self.continuous_topk(self.inject_noise(logits))

    def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()

        return samples

    def forward(self, logits):
        csamples = None
        if self.training:
            csamples = self.sample_continous(logits)

        dsamples = self.sample_discrete(logits)

        return dsamples, csamples


class DistilPredictor(nn.Module):
    def __init__(
        self,
        task_name: str,
        n_outs: int,
        voc_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 100,
        weights_matrix: Optional[torch.FloatTensor] = None,
        trainable_embeds: bool = False,
    ):
        super(DistilPredictor, self).__init__()

        self.lookup = create_emb_layer(weights_matrix, voc_size, embed_dim, trainable_embeds)
        embed_dim = self.lookup.embedding_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        if task_name == "reg":
            self.head = nn.Linear(hidden_dim, n_outs)
        elif task_name == "binary":
            self.head = nn.Sequential(nn.Linear(hidden_dim, n_outs), nn.Sigmoid())
        elif task_name == "multiclass":
            self.head = nn.Sequential(nn.Linear(hidden_dim, n_outs), nn.Softmax(dim=-1))

    def get_embedding(self, input):
        return self.lookup(input)

    def get_embedding_layer(self):
        return self.lookup

    def freeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(False)

    def unfreeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(True)

    def predict(self, embed, T):
        out = torch.mean(embed * T.unsqueeze(2), axis=1)
        out = self.act(self.fc1(out))
        out = self.head(out)

        return out

    def forward(self, x, T):
        embed = self.get_embedding(x)
        out = self.predict(embed, T)

        return out


class L2XModel(nn.Module):
    def __init__(
        self,
        task_name: str,
        n_outs: int,
        voc_size: int = 1000,
        embed_dim: int = 100,
        conv_filters: int = 100,
        conv_ksize: int = 3,
        drop_rate: float = 0.2,
        hidden_dim: int = 100,
        T: float = 0.3,
        k: int = 5,
        weights_matrix: Optional[torch.FloatTensor] = None,
        trainable_embeds: bool = False,
        sampler: str = "gumbeltopk",
        anneal_factor: float = 1.0,
    ):
        super(L2XModel, self).__init__()

        self.ti_model = TIModel(
            voc_size,
            embed_dim,
            conv_filters,
            conv_ksize,
            drop_rate,
            hidden_dim,
            weights_matrix,
            trainable_embeds,
        )  # token importance model
        self.T = T
        self.anneal_factor = anneal_factor
        if sampler == "gumbeltopk":
            self.sampler = GumbelTopKSampler(T, k)
        else:
            self.sampler = SoftSubSampler(T, k)
        self.distil_model = DistilPredictor(
            task_name,
            n_outs,
            voc_size,
            embed_dim,
            hidden_dim,
            weights_matrix,
            trainable_embeds,
        )

    def forward(self, x):
        logits = self.ti_model(x)
        dsamples, csamples = self.sampler(logits)
        if self.training:
            T = csamples
        else:
            T = dsamples
        out = self.distil_model(x, T)

        return out, T

    def anneal(self):
        self.sampler.T *= self.anneal_factor
