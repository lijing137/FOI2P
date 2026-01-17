import torch
import torch.nn as nn


class GroupNormlj(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNormlj, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0)  # (N, C) -> (1, N, C)
        x = x.transpose(1, 2)  # (B, N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, C, N) -> (B, N, C)
        if ndim == 2:
            x = x.squeeze(0)
        return x

    def __repr__(self):
        return self.norm.__repr__()


ACT_LAYERSlj = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    None: nn.Identity(),
}

class UnaryBlocklj(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm=32, activation_fn='leaky_relu', bias=True, layer_norm=False):
        super(UnaryBlocklj, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_fn = activation_fn
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNormlj(group_norm, out_channels)
        self.activation = ACT_LAYERSlj[activation_fn]

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x