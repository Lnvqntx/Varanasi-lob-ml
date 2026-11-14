"""Causal depthwise TCN, equivalent chunk and cached-row implementations.
All caches store each block's INPUT, chronologically oldest -> newest.
State tuple length 8, each (B,64,2*d); reset with None, detach only in trainer.
"""
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 64
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
RECEPTIVE_FIELD = 1 + 2 * sum(DILATIONS)
STATE_FLOATS = CHANNELS * 2 * sum(DILATIONS)


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(CHANNELS, CHANNELS, 3, dilation=dilation, groups=CHANNELS)
        self.pointwise = nn.Conv1d(CHANNELS, CHANNELS, 1)

    def forward(self, x, history):
        joined = torch.cat([history, x], dim=2)
        mixed = self.pointwise(F.relu(self.depthwise(joined)))
        return F.relu(x + mixed), joined[:, :, -2 * self.dilation:]

    def step(self, x, history):
        # Conv1d is cross-correlation: weights [0,1,2] use [t-2d,t-d,t].
        w = self.depthwise.weight[:, 0]
        temporal = (history[:, :, 0] * w[:, 0] + history[:, :, self.dilation] * w[:, 1]
                    + x * w[:, 2] + self.depthwise.bias)
        mixed = F.linear(F.relu(temporal), self.pointwise.weight[:, :, 0], self.pointwise.bias)
        return F.relu(x + mixed), torch.cat([history[:, :, 1:], x[:, :, None]], 2)


class CausalTCN(nn.Module):
    def __init__(self, n_in=168, hidden=128, layers=2, n_out=2):
        super().__init__()
        assert n_in == 168 and n_out == 2
        # hidden/layers are the shared trainer's GRU settings, not TCN channel/depth knobs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(34_000_000 + torch.initial_seed())
            self.projection = nn.Linear(n_in, CHANNELS)
            self.blocks = nn.ModuleList([TemporalBlock(d) for d in DILATIONS])
            self.head = nn.Linear(CHANNELS, n_out)

    def initial_state(self, x):
        return tuple(x.new_zeros(x.shape[0], CHANNELS, 2*d) for d in DILATIONS)

    def forward(self, x, state=None):
        if state is None:
            state = self.initial_state(x)
        x = F.relu(self.projection(x)).transpose(1, 2)
        new_state = []
        for block, history in zip(self.blocks, state):
            x, history = block(x, history)
            new_state.append(history)
        return self.head(x.transpose(1, 2)), tuple(new_state)

    def step(self, x, state=None):
        """x (B,168), output (B,2); identical state layout to batched forward."""
        if state is None:
            state = self.initial_state(x)
        x = F.relu(self.projection(x))
        new_state = []
        for block, history in zip(self.blocks, state):
            x, history = block.step(x, history)
            new_state.append(history)
        return self.head(x), tuple(new_state)


def register():
    import train_multi
    train_multi.ARCHS['v34tcn'] = CausalTCN
    return train_multi
