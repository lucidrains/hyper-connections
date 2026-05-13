import torch
import torch.nn.functional as F
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
        x, packed_shape = pack([x], '* d')
        residual, _ = pack([residual], '* d')

        output = self.gru(x, residual)

        output, = unpack(output, packed_shape, '* d')
        return output

class GatedResidual(Module):
    def __init__(
        self,
        dim,
        fine_gate = False
    ):
        super().__init__()

        self.to_learned_mix = nn.Linear(dim * 2, dim if fine_gate else 1)

    def forward(self, x, residual):
        x_and_residual, _ = pack([x, residual], 'b n *')

        mix = self.to_learned_mix(x_and_residual)

        out = x.lerp(residual, mix.sigmoid())
        return out

class OrthogonalResidualUpdate(Module):
    def __init__(
        self,
        double_precision = True
    ):
        super().__init__()
        self.double_precision = double_precision

    def forward(self, x, residual):
        use_double, dtype = self.double_precision, residual.dtype

        if use_double:
            residual, x = residual.double(), x.double()

        unit = F.normalize(residual, dim = -1)
        parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
        orthogonal = x - parallel

        if use_double:
            orthogonal = orthogonal.to(dtype)

        return residual + orthogonal

class MVSplitResidualUpdate(Module):
    # Pengqi Lu https://arxiv.org/abs/2605.06169

    def __init__(
        self,
        dim,
        init_alpha = 0.,
        init_beta = 0.03,
        **kwargs
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim) * init_alpha)
        self.beta = nn.Parameter(torch.ones(dim) * init_beta)

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        mean_x = x.mean(dim = -2, keepdim = True)
        mean_residual = residual.mean(dim = -2, keepdim = True)

        centered_x = x - mean_x

        return residual + centered_x * self.beta + (mean_x - mean_residual) * self.alpha
