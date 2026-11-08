"""Learned causal multiscale context. State is (GRU hidden, B x 3 x 32 EMA).
EMA blocks never exceed 64 rows: no inverse powers or long-sequence products.
Only the trainer detaches state, after each optimizer step.
"""
import math
import torch
from torch import nn

HALF_LIVES = (8, 64, 512)


def averages(z, state=None):
    """Inclusive EMA; initialise every scale to z[0]. Differentiable prefix scan."""
    alpha = z.new_tensor([-math.expm1(-math.log(2) / h) for h in HALF_LIVES])[None, None, :, None]
    decay = 1 - alpha
    fresh = state is None
    if fresh:
        state = z[:, 0, None, :].expand(-1, 3, -1)
    blocks = []
    for start in range(0, z.shape[1], 64):
        b = alpha * z[:, start:start + 64, None, :]
        first = state[:, None] if fresh and start == 0 else b[:, :1] + decay * state[:, None]
        b = torch.cat([first, b[:, 1:]], 1)
        offset = 1
        while offset < b.shape[1]:
            b = torch.cat([b[:, :offset], b[:, offset:] + decay.pow(offset) * b[:, :-offset]], 1)
            offset *= 2
        blocks.append(b)
        state = b[:, -1]
    return torch.cat(blocks, 1), state


class ProjectedGRU(nn.Module):
    context = False

    def __init__(self, n_in=168, hidden=128, layers=2, n_out=2):
        super().__init__()
        # Own projection RNG; B and C have exactly the same projection at init.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33_000_000 + torch.initial_seed())
            self.projection = nn.Linear(n_in, 32)
        self.gru = nn.GRU(128 if self.context else 32, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, state=None):
        z = torch.tanh(self.projection(x))
        h, ema = (None, None) if state is None else state
        if self.context:
            av, ema = averages(z, ema)
            features = torch.cat([z, (z[:, :, None] - av).flatten(2)], 2)
        else:
            features = z
            ema = z.new_zeros(z.shape[0], 3, 32)
        out, h = self.gru(features, h)
        return self.head(out), (h, ema)


class ContextGRU(ProjectedGRU):
    context = True


def register():
    import train_multi
    train_multi.ARCHS.update(v33proj=ProjectedGRU, v33context=ContextGRU)
    return train_multi
