from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Reduce, Rearrange

from hyper_connections.hyper_connections import (
    Residual,
    RMSNorm
)

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
"""

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# main functions

def get_expand_reduce_stream_functions(num_streams, disable = False):

    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)
    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(num_streams, disable = None):

    disable = default(disable, num_streams == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, disable = disable)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * (self.gamma + 1)

# hyper connection residual streams

class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        tanh = True,
        channel_first = True,
        dropout = 0.,
        residual_transform: Module | None = None, # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch

        # activation, seemingly results were wishy washy depending on using tanh or not

        self.act = nn.Tanh() if tanh else nn.Identity()

        self.norm = RMSNorm(dim) # they used layernorm in paper, but rmsnorm is fine given what we know now

        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams # just choose one random residual stream if layer index not given

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, 1))
        init_alpha0[init_residual_index, 0] = 1.

        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Conv2d(dim, num_residual_streams + 1, 1, bias = False)
        nn.init.zeros_(self.dynamic_alpha_fn.weight)

        self.dynamic_beta_fn = nn.Sequential(
            nn.Conv2d(dim, 1, 1, bias = False),
            Rearrange('b 1 ... -> b ...')
        )

        nn.init.zeros_(self.dynamic_beta_fn[0].weight)

        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)


        # dropouts

        self.dropout = nn.Dropout(dropout)

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        normed = self.norm(residuals)

        # alpha for weighted sum of residuals going into branch

        wc_weight = self.act(self.dynamic_alpha_fn(normed))
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale

        dynamic_alpha = rearrange(dynamic_alpha, '(b s) ... -> b s ...', s = self.num_residual_streams)
        alpha = dynamic_alpha + rearrange(self.static_alpha, 's t -> s t 1 1')

        # beta for weights from branch output back to residual streams

        dc_weight = self.act(self.dynamic_beta_fn(normed))
        dynamic_beta = dc_weight * self.dynamic_beta_scale        
        dynamic_beta = rearrange(dynamic_beta, '(b s) ... -> b s ...', s = self.num_residual_streams)
        beta = dynamic_beta + rearrange(self.static_beta, 's -> s 1 1')

        residuals = rearrange(residuals, '(b s) ... -> b s ...', s = self.num_residual_streams)
        mix_h = einsum(alpha, residuals, 'b s t ..., b s d ... -> b t d ...')

        branch_input, residuals = mix_h[:, 0, ...], mix_h[:, 1:, ...]

        return branch_input, maybe_transformed_residuals, dict(beta = beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        # 'depth' connection

        output = einsum(branch_output, beta, 'b d ..., b s ... -> b s d ...')
        output = rearrange(output, 'b s d ... -> (b s) d ...')

        residuals = residuals + output

        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

HyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
