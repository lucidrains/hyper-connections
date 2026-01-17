from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
p - proposals
"""

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y

# sinkhorn

def l1norm(t, dim):
    return F.normalize(t, p = 1, dim = dim)

def sinkhorn_knopps(log_alpha, iters = 20):

    if iters <= 0:
        return log_alpha

    assert log_alpha.shape[-2] == log_alpha.shape[-1]

    dtype = log_alpha.dtype
    log_alpha = log_alpha.float()

    log_alpha = log_alpha - log_alpha.amax(dim = -2, keepdim = True).detach()

    alpha = log_alpha.exp()

    for _ in range(iters):
        alpha = l1norm(alpha, dim = -2)
        alpha = l1norm(alpha, dim = -1)

    return alpha.to(dtype)

def log_domain_sinkhorn_knopps(log_alpha, iters = 20):

    if iters <= 0:
        return log_alpha

    assert log_alpha.shape[-2] == log_alpha.shape[-1]

    dtype = log_alpha.dtype
    log_alpha = log_alpha.float()

    for _ in range(iters):
        log_alpha = F.log_softmax(log_alpha, dim = -2)
        log_alpha = F.log_softmax(log_alpha, dim = -1)

    return log_alpha.exp().to(dtype)

# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    add_attn_pool_reduce_stream = False,
    dim = None,
    disable = False
):
    if disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed or add_attn_pool_reduce_stream:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

    if add_stream_embed:
        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams = True)
    else:
        expand_fn = Reduce('... d -> ... s d', 'repeat', s = num_streams)

    if add_attn_pool_reduce_stream:
        reduce_fn = AttentionPoolReduceStream(dim)
    else:
        reduce_fn = Reduce('... s d -> ... d', 'sum')

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    add_attn_pool_reduce_stream = False,
    disable = None,
    sinkhorn_iters = 20,
    use_triton_sinkhorn = False,
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = ManifoldConstrainedHyperConnections if not disable else Residual

    kwargs.pop('add_attn_pool_reduce_stream', None)
    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, sinkhorn_iters = sinkhorn_iters, use_triton_sinkhorn = use_triton_sinkhorn, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams,
        add_stream_embed = add_stream_embed,
        add_attn_pool_reduce_stream = add_attn_pool_reduce_stream,
        dim = dim,
        disable = disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals
    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# hyper connection residual streams

class ManifoldConstrainedHyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        dropout = 0.,
        residual_transform: Module | None = None, # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual = True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views = 1,                # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn = add,
        num_fracs = 1,                      # https://arxiv.org/abs/2503.14125
        sinkhorn_iters = 20,
        log_domain_sinkhorn = False,
        residual_mix_constraint_fn: Callable | None = None,
        forward_method_names: tuple[str, ...] = (),
        num_dynamic_alpha_proposals = 1,
        use_triton_sinkhorn = False,
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f = num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')

        assert divisible_by(dim, num_fracs), f'feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})'

        dim //= num_fracs # effective dim handled in dimension is feature dimension divided by num fractions

        # they used layernorm in paper, but rmsnorm is fine given what we know now

        self.norm = RMSNorm(dim)

        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams # just choose one random residual stream if layer index not given

        # handle the parameter dimensions, which may require (num_residuals x num_fractions) - generalizing hyper + frac connections

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs

        # width num residual streams

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # number of dynamic alpha proposals, for averaging Hres across proposals

        self.has_dynamic_alpha_proposals = num_dynamic_alpha_proposals > 1
        self.num_dynamic_alpha_proposals = num_dynamic_alpha_proposals

        # width connection

        init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
        init_alpha0[init_residual_index, :] = 1.

        self.static_alpha = nn.Parameter(cat((init_alpha0, torch.eye(num_residual_streams_fracs)), dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(num_dynamic_alpha_proposals, dim, num_residual_streams_fracs + num_input_views_fracs))

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            self.static_beta = nn.Parameter(torch.ones(num_residual_streams, num_fracs, 1))

            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim, num_fracs))

            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # Hres constraint related
        # by default is sinkhorn

        use_triton_sinkhorn_and_available = False

        if use_triton_sinkhorn:
            try:
                from hyper_connections.triton_sinkhorn import triton_sinkhorn, is_triton_available
                use_triton_sinkhorn_and_available = is_triton_available()
            except ImportError:
                use_triton_sinkhorn_and_available = False

        if use_triton_sinkhorn_and_available:
            self.residual_mix_constraint_fn = partial(triton_sinkhorn, iters = sinkhorn_iters)
        else:
            self.residual_mix_constraint_fn = default(
                residual_mix_constraint_fn,
                partial(sinkhorn_knopps if not log_domain_sinkhorn else log_domain_sinkhorn_knopps, iters = sinkhorn_iters)
            )

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

        # forwarding method names

        self.forward_method_names = forward_method_names

        for forward_method_name in self.forward_method_names:
            assert not hasattr(self, forward_method_name)

            fn = getattr(self.branch, forward_method_name)
            setattr(self, forward_method_name, fn)

    def width_connection(
        self,
        residuals
    ):
        streams, fracs = self.num_residual_streams, self.num_fracs

        residuals = self.residual_transform(residuals)

        # width connection

        # split out fractions

        residuals = self.split_fracs(residuals)

        # norm

        normed = self.norm(residuals)

        # alpha for weighted sum of residuals going into branch

        dtype = residuals.dtype

        normed = normed.float()

        wc_weight = einsum(normed, self.dynamic_alpha_fn.float(), '... d, p d mix -> p ... mix')
        wc_weight = rearrange(wc_weight, '... s1 f2 mix -> ... (s1 f2) mix')

        pre_branch_scale = repeat(self.pre_branch_scale.float(), '1 -> s', s = self.num_fracs)
        residual_scale = repeat(self.residual_scale.float(), '1 -> s', s = self.num_fracs * streams)
        alpha_scale = cat((pre_branch_scale, residual_scale))

        alpha_scale = repeat(alpha_scale, 'n -> (v n)', v = self.num_input_views)

        dynamic_alpha = wc_weight * alpha_scale

        alpha = dynamic_alpha + self.static_alpha.float()

        # the alpha is now split and "manifold constrained" with sinkhorn and sigmoid

        alpha_pre, alpha_residual = alpha[..., :self.num_input_views * self.num_fracs], alpha[..., self.num_input_views * self.num_fracs:]

        alpha_pre = alpha_pre.sigmoid()

        alpha_residual = self.residual_mix_constraint_fn(alpha_residual)

        alpha = cat((alpha_pre, alpha_residual), dim = -1)

        if self.has_dynamic_alpha_proposals:
            alpha = reduce(alpha, 'p ... -> ...', 'mean')
        else:
            alpha = rearrange(alpha, '1 ... -> ...')

        alpha = rearrange(alpha, '... (s f) t -> ... s f t', s = streams) # (batch, seq, fracs1, streams, fracs2, input + residual streams)

        # beta for weights from branch output back to residual streams

        beta = None

        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn.float()

            dynamic_beta = dc_weight * self.h_post_scale.float()

            beta = dynamic_beta + self.static_beta.float()

            beta = beta.sigmoid() * 2 # for "H_post" manifold constraint

        mix_h = einsum(alpha, residuals.float(), '... s f tf, ... s f d -> ... tf d')

        mix_h = rearrange(mix_h, '... (t f) d -> ... t f d', f = fracs)

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :, :], mix_h[..., 1:, :, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :, :], mix_h[..., self.num_input_views:, :, :]
            branch_input = rearrange(branch_input, 'b ... v f d -> v b ... f d')

        # maybe merge fractions back

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, 'b ... s f d -> b ... s (f d)')

        branch_input, residuals = tuple(t.to(dtype) for t in (branch_input, residuals))

        if exists(beta):
            beta = beta.to(dtype)

        return branch_input, residuals, dict(beta = beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta
    ):
        assert self.add_branch_out_to_residual

        # maybe split fractions

        branch_output = self.split_fracs(branch_output)

        # 'depth' connection

        dtype = residuals.dtype

        output = einsum(branch_output.float(), beta.float(), 'b ... f1 d, b ... s f1 f2 -> b ... s f2 d')

        # merge merge back fractions

        output = self.merge_fracs(output)

        # channel first

        residuals = self.depth_residual_fn(output.to(dtype), residuals)

        return self.dropout(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):

            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

mHC = ManifoldConstrainedHyperConnections

ManifoldConstrainedHyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
ManifoldConstrainedHyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# stream embed

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        expand_to_streams = False
    ):
        super().__init__()
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, '... d -> ... s d', s = self.num_streams)

        return residuals + self.stream_embed

# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else

class AttentionPoolReduceStream(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        return einsum(residuals, attn, '... s d, ... s d -> ... d')
