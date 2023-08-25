import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/lucidrains/tab-transformer-pytorch/tree/main
# feedforward and attention
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = self.heads
        
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, qkv.size(-1) // 3, dim=-1)
        
        q = q.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)
        
        q = q * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(sim, dim=-1)
        dropped_attn = self.dropout(attn)
        
        out = torch.matmul(dropped_attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        return out, attn

# transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        return_attn = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.return_attn = return_attn

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = attn_out + x
            x = ff(x) + x

        if not self.return_attn:
            return x

        return x, torch.stack(post_softmax_attns)