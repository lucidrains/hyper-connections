import torch
from torch import nn
from torch.nn import Module

from einops import rearrange, pack, unpack

class GRUGatedResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        x, ps = pack([x], '* d')
        residual, _ = pack([residual], '* d')

        output = self.gru(x, residual)

        output, = unpack(output, ps, '* d')
        return output

class GatedResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.to_learned_mix = nn.Linear(dim * 2, dim)

    def forward(self, x, residual):
        x_and_residual, _ = pack([x, residual], 'b n *')

        mix = self.to_learned_mix(x_and_residual)

        out = x.lerp(residual, mix.sigmoid())
        return out
